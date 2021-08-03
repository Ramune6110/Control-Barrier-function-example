clear;
close all;
clc;

% Init state.
x0 = [0; 0; 4; 0];

% Target position
params.p_d = [10; 0];
% obstacle center
params.p_o = [5; 2];
% obstacle radius.
params.r_o = 2; 

dt = 0.02;
sim_t = 30;

params.cbf_gamma0 = 1;

params.u_max = 7;
params.u_min  = -7;

params.clf.rate = 0.7;
params.cbf.rate = 3;

params.weight.slack = 1;
params.weight.input = 5;

[x, f, g] = defineSystem();
cbf = defineCbf(params, x);
obj = initSys(x, f, g, cbf);

total_k = ceil(sim_t / dt);
x = x0;
t = 0;   
% initialize traces.
xs = zeros(total_k, obj.xdim);
ts = zeros(total_k, 1);
us = zeros(total_k-1, obj.udim);
hs = zeros(total_k-1, 1);
Vs = zeros(total_k-1, 1);
xs(1, :) = x0';
ts(1) = t;
u_prev = [0;0];
for k = 1:total_k-1
    t
    % Determine control input.
    % dV_hat: analytic Vdot based on model.
    [u, slack, h, V] = ctrlCbfQp(obj, params, x, 1);  
%     [u, slack, h, V] = controller(s, u_prev); % optimizing the difference between the previous timestep.       
    us(k, :) = u';
    hs(k) = h;
    Vs(k) = V;

    % Run one time step propagation.
    [ts_temp, xs_temp] = ode45(@(t, s) dynamics(obj, t, s, u), [t t+dt], x);
    x = xs_temp(end, :)';

    xs(k+1, :) = x';
    ts(k+1) = ts_temp(end);
    u_prev = u;
    t = t + dt;
end

plot_results(ts, xs, us, params.p_o, params.r_o)

function [x, f, g] = defineSystem()
    syms p_x v_x p_y v_y;
    x = [p_x; v_x; p_y; v_y];

    A = zeros(4);
    A(1, 2) = 1;
    A(3, 4) = 1;
    B = [0 0; 1 0; 0 0; 0 1];

    f = A * x;
    g = B;
end

function dx = dynamics(obj, t, x, u)
    % Inputs: t: time, x: state, u: control input
    % Output: dx: \dot(x)
    dx = obj.f(x) + obj.g(x) * u;
end

function cbf = defineCbf(params, symbolic_state)
    x = symbolic_state;
    p_o = params.p_o; % position of the obstacle.
    r_o = params.r_o; % radius of the obstacle.
    
    distance = (x(1) - p_o(1))^2 + (x(3) - p_o(2))^2 - r_o^2;
    derivDistance = 2*(x(1)-p_o(1))*x(2) + 2*(x(3)-p_o(2))*x(4);
    cbf = derivDistance + params.cbf_gamma0 * distance;
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

function [u, B, feas, comp_time] = ctrlCbfQp(obj, params, x, verbose)
    %% Implementation of vanilla CBF-QP
    % Inputs:   x: state
    %           u_ref: reference control input
    %           verbose: flag for logging (1: print log, 0: run silently)
    % Outputs:  u: control input as a solution of the CBF-CLF-QP
    %           B: Value of the CBF at current state.
    %           feas: 1 if QP is feasible, 0 if infeasible. (Note: even
    %           when qp is infeasible, u is determined from quadprog.)
    %           compt_time: computation time to run the solver.  
    u_ref = zeros(obj.udim, 1);
    
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
%     H = (x(1) - params.xd)^2 + (x(2) - params.yd)^2;
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

function plot_results(t, xs, us, p_o, r_o)

figure
subplot(5,1,1)
plot(t, xs(:,1))
xlabel('t')
ylabel('x [m]')

subplot(5,1,2)
plot(t, xs(:,2))
xlabel('t')
ylabel('v_x [m/s]')

subplot(5,1,3)
plot(t, xs(:,3))
xlabel('t')
ylabel('y [m]')

subplot(5,1,4)
plot(t, xs(:, 4))
xlabel('t')
ylabel('v_y [m/s]')

subplot(5,1,5)
plot(t, sqrt(xs(:, 2).^2 + xs(:, 4).^2))
xlabel('t')
ylabel('v [m/s]')


figure
subplot(2,1,1)
plot(t(1:end-1), us(:,1))
xlabel('t')
ylabel('a_x [m/s^2]')

subplot(2,1,2)
plot(t(1:end-1), us(:,2))
xlabel('t')
ylabel('a_y [m/s^2]')


lim_min = min(min(xs(:, 1)), min(xs(:, 3)));
lim_max = max(max(xs(:, 1)), max(xs(:, 3)));
lim_min = min([lim_min, p_o(1)-r_o, p_o(2)-r_o]);
lim_max = max([lim_max, p_o(1)+r_o, p_o(2)+r_o]);

figure
plot(xs(:, 1), xs(:, 3));
draw_circle(p_o, r_o);

xlim([lim_min, lim_max]);
ylim([lim_min, lim_max]);
xlabel('x [m]')
ylabel('y [m]')
end

function h = draw_circle(center,r)
hold on
th = 0:pi/50:2*pi;
xunit = r * cos(th) + center(1);
yunit = r * sin(th) + center(2);
h = plot(xunit, yunit);
hold off
end