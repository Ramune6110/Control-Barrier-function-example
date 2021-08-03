clear;
close all;
clc;

dt = 0.02;
sim_t = 20;
x0 = [0; 20; 100];

%% Parameters are from 
% Aaron Ames et al. Control Barrier Function based Quadratic Programs 
% with Application to Adaptive Cruise Control, CDC 2014, Table 1.

params.v0 = 14;
params.vd = 24;
params.m  = 1650;
params.g = 9.81;
params.f0 = 0.1;
params.f1 = 5;
params.f2 = 0.25;
params.ca = 0.3;
params.cd = 0.3;
params.T = 1.8;

params.u_max = params.ca * params.m * params.g;
params.u_min  = -params.cd * params.m * params.g;

params.clf.rate = 5;
params.cbf.rate = 5;

params.weight.input = 2/params.m^2;
params.weight.slack = 2e-2;

[x, f, g] = defineSystem(params);
cbf = defineCbf(params, x);
obj = initSys(x, f, g, cbf);

total_k = ceil(sim_t / dt);
x = x0;
t = 0;   
% initialize traces.
xs = zeros(total_k, 3);
ts = zeros(total_k, 1);
us = zeros(total_k-1, 1);
slacks = zeros(total_k-1, 1);
hs = zeros(total_k-1, 1);
Vs = zeros(total_k-1, 1);
xs(1, :) = x0';
ts(1) = t;

for k = 1:total_k-1
    t
    Fr = getFr(x, params);
    % Determine control input.
    [u, slack, h, V] = ctrlCbfQp(obj, params, x, Fr, 0);        
    us(k, :) = u';
    slacks(k, :) = slack;
    hs(k) = h;
    Vs(k) = V;

    % Run one time step propagation.
    [ts_temp, xs_temp] = ode45(@(t, s) dynamics(obj, t, s, u), [t t+dt], x);
    x = xs_temp(end, :)';

    xs(k+1, :) = x';
    ts(k+1) = ts_temp(end);
    t = t + dt;
end

plot_results(ts, xs, us, slacks, hs, Vs, params)

function [x, f, g] = defineSystem(params)
    syms p v z % states
    x = [p; v; z];

    f0 = params.f0;
    f1 = params.f1;
    f2 = params.f2;
    v0 = params.v0;
    m = params.m;
    Fr = f0 + f1 * v + f2 * v^2;

    % Dynamics
    f = [v; -Fr/m; v0-v];
    g = [0; 1/m; 0]; 
end

function cbf = defineCbf(params, symbolic_state)
    v = symbolic_state(2);
    z = symbolic_state(3);

    v0 = params.v0;
    T = params.T;
    cd = params.cd;
    
    cbf = z - T * v - 0.5  * (v0-v)^2 / (cd * params.g);
end

function obj = initSys(symbolic_x, symbolic_f, symbolic_g, symbolic_cbf)
    if isempty(symbolic_x) || isempty(symbolic_f) || isempty(symbolic_g)
        error('x, f, g is empty. Create a class function defineSystem and define your dynamics with symbolic expression.');
    end

    if ~isa(symbolic_f, 'sym')
        f_ = sym(symbolic_f);
    else
        f_ = symbolic_f;
    end
    if ~isa(symbolic_g, 'sym')
        g_ = sym(symbolic_g);
    else
        g_ = symbolic_g;
    end
    
    x = symbolic_x;
    % Setting state and input dimension.
    obj.xdim = size(x, 1);
    obj.udim = size(g_, 2);
    % Setting f and g (dynamics)
    obj.f = matlabFunction(f_, 'vars', {x});
    obj.g = matlabFunction(g_, 'vars', {x});            

    % Obtaining Lie derivatives of CBF.
    if ~isempty(symbolic_cbf)
        dcbf = simplify(jacobian(symbolic_cbf, symbolic_x));
        lf_cbf_ = dcbf * f_;
        lg_cbf_ = dcbf * g_;        
        obj.cbf = matlabFunction(symbolic_cbf, 'vars', {x});
        obj.lf_cbf = matlabFunction(lf_cbf_, 'vars', {x});
        % TODO: add sanity check of relative degree.
        obj.lg_cbf = matlabFunction(lg_cbf_, 'vars', {x});
    end
end

function dx = dynamics(obj, t, x, u)
    % Inputs: t: time, x: state, u: control input
    % Output: dx: \dot(x)
    dx = obj.f(x) + obj.g(x) * u;
end

function Fr = getFr(x, params)
    v = x(2);
    Fr = params.f0 + params.f1 * v + params.f2 * v^2;
end
        
%% Author: Jason Choi (jason.choi@berkeley.edu)
function [u, B, feas, comp_time] = ctrlCbfQp(obj, params, x, u_ref, verbose)
    %% Implementation of vanilla CBF-QP
    % Inputs:   x: state
    %           u_ref: reference control input
    %           verbose: flag for logging (1: print log, 0: run silently)
    % Outputs:  u: control input as a solution of the CBF-CLF-QP
    %           B: Value of the CBF at current state.
    %           feas: 1 if QP is feasible, 0 if infeasible. (Note: even
    %           when qp is infeasible, u is determined from quadprog.)
    %           compt_time: computation time to run the solver.
    if isempty(obj.cbf)
        error('CBF is not defined so ctrlCbfQp cannot be used. Create a class function [defineCbf] and set up cbf with symbolic expression.');
    end
        
    if nargin < 3
        u_ref = zeros(obj.udim, 1);
    end
    if nargin < 4
        % Run QP without log in default condition.
        verbose = 0;
    end

    if size(u_ref, 1) ~= obj.udim
        error("Wrong size of u_ref, it should be (udim, 1) array.");
    end                
            
    tstart = tic;
    B = obj.cbf(x);
    LfB = obj.lf_cbf(x);
    LgB = obj.lg_cbf(x);
        
    %% Constraints : A * u <= b
    % CBF constraint.
    A = [-LgB];
    b = [LfB + params.cbf.rate * B];                
    % Add input constraints if u_max or u_min exists.
    if isfield(params, 'u_max')
        A = [A; eye(obj.udim)];
        if size(params.u_max, 1) == 1
            b = [b; params.u_max * ones(obj.udim, 1)];
        elseif size(params.u_max, 1) == obj.udim
            b = [b; params.u_max];
        else
            error("params.u_max should be either a scalar value or an (udim, 1) array.")
        end
    end
    if isfield(params, 'u_min')
        A = [A; -eye(obj.udim)];
        if size(params.u_min, 1) == 1
            b = [b; -params.u_min * ones(obj.udim, 1)];
        elseif size(params.u_min, 1) == obj.udim
            b = [b; -params.u_min];
        else
            error("params.u_min should be either a scalar value or an (udim, 1) array")
        end
    end

    %% Cost
    if isfield(params.weight, 'input')
        if size(params.weight.input, 1) == 1 
            weight_input = params.weight.input * eye(obj.udim);
        elseif all(size(params.weight.input) == obj.udim)
            weight_input = params.weight.input;
        else
            error("params.weight.input should be either a scalar value or an (udim, udim) array.")
        end
    else
        weight_input = eye(obj.udim);
    end
    
    if verbose
        options =  optimset('Display','notify');
    else
        options =  optimset('Display','off');
    end

    % cost = 0.5 u' H u + f u
    H = weight_input;
    f_ = -weight_input * u_ref;
    [u, ~, exitflag, ~] = quadprog(H, f_, A, b, [], [], [], [], [], options);
    if exitflag == -2
        feas = 0;
        disp("Infeasible QP. CBF constraint is conflicting with input constraints.");
    else
        feas = 1;
    end
    comp_time = toc(tstart);
end

function plot_results(ts, xs, us, slacks, hs, Vs, params)
    fig_sz = [10 15]; 
    plot_pos = [0 0 10 15];
    yellow = [0.998, 0.875, 0.529];
    blue = [0.106, 0.588, 0.953];
    navy = [0.063, 0.075, 0.227];
    magenta = [0.937, 0.004, 0.584];
    orange = [0.965, 0.529, 0.255];
    grey = 0.01 *[19.6, 18.8, 19.2];
    
    figure(1);
    subplot(4,1,1);
    p = plot(ts, xs(:, 2));
    p.Color = blue;
    p.LineWidth = 1.5;
    hold on;
    plot(ts, params.vd*ones(size(ts, 1), 1), 'k--');
    ylabel("v (m/s)");
    title("State - Velocity");
    set(gca,'FontSize',14);
    grid on;    
    
    subplot(4,1,2);
    p = plot(ts, xs(:, 3));
    p.Color = magenta;
    p.LineWidth = 1.5;
    ylabel("z (m)");
    title("State - Distance to lead vehicle");
    set(gca, 'FontSize', 14);
    grid on;    
    
    subplot(4,1,3);
    p = plot(ts(1:end-1), us); hold on;
    p.Color = orange;
    p.LineWidth = 1.5;
    plot(ts(1:end-1), params.u_max*ones(size(ts, 1)-1, 1), 'k--');
    plot(ts(1:end-1), params.u_min*ones(size(ts, 1)-1, 1), 'k--');
    ylabel("u(N)");
    title("Control Input - Wheel Force");    
    set(gca, 'FontSize', 14);
    grid on;    

    subplot(4,1,4);
    p = plot(ts(1:end-1), slacks); hold on;
    p.Color = magenta;
    p.LineWidth = 1.5;
    ylabel("CBF");
    title("CBF variable");        
    set(gca, 'FontSize', 14);
    grid on;        
end