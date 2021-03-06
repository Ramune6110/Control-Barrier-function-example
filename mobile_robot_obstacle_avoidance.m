clear;
close all;
clc;

dt = 0.02;
sim_t = 20;
x0 = [0;5;0];

params.v = 1; % velocity
params.u_max = 3; % max yaw rate (left)
params.u_min = -3; % min yaw rate (right)

% Obstacle position
params.xo = 5;
params.yo = 4;
% Obstacle radius
params.d = 2;
params.cbf_gamma0 = 1;
% Desired target point
params.xd = 20;
params.yd = 5;

params.clf.rate = 0.5;
params.weight.slack = 10;

params.cbf.rate = 10;

[x, f, g] = defineSystem(params);
cbf = defineCbf(params, x);
clf = defineClf(params, x);
obj = initSys(x, f, g, cbf, clf);

total_k = ceil(sim_t / dt);
x = x0;
t = 0;   
% initialize traces.
xs = zeros(total_k, 3);
ts = zeros(total_k, 1);
us = zeros(total_k-1, 1);
Vs = zeros(total_k-1, 1);
hs = zeros(total_k-1, 1);
xs(1, :) = x0';
ts(1) = t;
for k = 1:total_k-1
    t
    % Determine control input.
    % dV_hat: analytic Vdot based on model.
%     [u, slack, h, V] = ctrlCbfQp(obj, params, x, 1);       
    [u, slack, h, V, feas, comp_time] = ctrlCbfClfQp(obj, params, x, 1, 1);   
    us(k, :) = u';
    hs(k) = h;
    Vs(k) = V;

    % Run one time step propagation.
    [ts_temp, xs_temp] = ode45(@(t, s) dynamics(obj, t, s, u), [t t+dt], x);
    
    x = xs_temp(end, :)';

    xs(k+1, :) = x';
    ts(k+1) = ts_temp(end);
    t = t + dt;
end

plot_results(ts, xs, us, hs, [params.xo;params.yo], params.d)

function [x, f, g] = defineSystem(params)
    syms p_x p_y theta
    x = [p_x; p_y; theta];
    f = [params.v * cos(theta);
        params.v * sin(theta);
        0];
    g = [0; 0; 1];
end

function dx = dynamics(obj, t, x, u)
    % Inputs: t: time, x: state, u: control input
    % Output: dx: \dot(x)
    dx = obj.f(x) + obj.g(x) * u;
end

function cbf = defineCbf(params, symbolic_state)
    x = symbolic_state;
    p_x = x(1); p_y = x(2); theta = x(3);

    v = params.v;
    xo = params.xo;
    yo = params.yo;
    d = params.d;

    distance = (p_x - xo)^2 + (p_y - yo)^2 - d^2;
    derivDistance = 2*(p_x-xo)*v*cos(theta) + 2*(p_y-yo)*v*sin(theta);
    cbf = derivDistance + params.cbf_gamma0 * distance; 
end

function clf = defineClf(params, symbolic_state)
    p_x = symbolic_state(1);
    p_y = symbolic_state(2);
    theta = symbolic_state(3);
    clf = (cos(theta).*(p_y-params.yd)-sin(theta).*(p_x-params.xd)).^2;
end

function obj = initSys(symbolic_x, symbolic_f, symbolic_g, symbolic_cbf, symbolic_clf)
    % Dynamics
    f_ = sym(symbolic_f);
    g_ = sym(symbolic_g);
    
    x = symbolic_x;
    % Setting state and input dimension.
    obj.xdim = size(x, 1);
    obj.udim = size(g_, 2);
    
    % Setting f and g (dynamics)
    obj.f = matlabFunction(f_, 'vars', {x});
    obj.g = matlabFunction(g_, 'vars', {x});            

    % Obtaining Lie derivatives of CBF.
    dcbf = simplify(jacobian(symbolic_cbf, symbolic_x));
    lf_cbf_ = dcbf * f_;
    lg_cbf_ = dcbf * g_;        
    obj.cbf = matlabFunction(symbolic_cbf, 'vars', {x});
    obj.lf_cbf = matlabFunction(lf_cbf_, 'vars', {x});
    % TODO: add sanity check of relative degree.
    obj.lg_cbf = matlabFunction(lg_cbf_, 'vars', {x});
   
    % Obtaining Lie derivatives of CLF.    
    dclf = simplify(jacobian(symbolic_clf, symbolic_x));
    lf_clf_ = dclf * f_;
    lg_clf_ = dclf * g_;
    obj.clf = matlabFunction(symbolic_clf, 'vars', {x});                       
    obj.lf_clf = matlabFunction(lf_clf_, 'vars', {x});
    % TODO: add sanity check of relative degree.
    obj.lg_clf = matlabFunction(lg_clf_, 'vars', {x});        
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

%% Author: Jason Choi (jason.choi@berkeley.edu)
function [u, slack, B, V, feas, comp_time] = ctrlCbfClfQp(obj, params, x, verbose, with_slack)
    %% Implementation of vanilla CBF-CLF-QP
    % Inputs:   x: state
    %           u_ref: reference control input
    %           with_slack: flag for relaxing the clf constraint(1: relax, 0: hard-constraint)
    %           verbose: flag for logging (1: print log, 0: run silently)
    % Outputs:  u: control input as a solution of the CBF-CLF-QP
    %           slack: slack variable for relaxation. (empty list when with_slack=0)
    %           B: Value of the CBF at current state.
    %           V: Value of the CLF at current state.
    %           feas: 1 if QP is feasible, 0 if infeasible. (Note: even
    %           when qp is infeasible, u is determined from quadprog.)
    %           comp_time: computation time to run the solver.
      
    u_ref = zeros(obj.udim, 1);
    
    tstart = tic;
    V = obj.clf(x);
    LfV = obj.lf_clf(x);
    LgV = obj.lg_clf(x);

    B = obj.cbf(x);
    LfB = obj.lf_cbf(x);
    LgB = obj.lg_cbf(x);
    
    %% Constraints: A[u; slack] <= b
    if with_slack
        % CLF and CBF constraints.
        A = [LgV, -1;
            -LgB, 0];
        b = [-LfV - params.clf.rate * V;
             LfB + params.cbf.rate * B];                
        % Add input constraints if u_max or u_min exists.
        if isfield(params, 'u_max')
            A = [A; eye(obj.udim), zeros(obj.udim, 1);];
            if size(params.u_max, 1) == 1
                b = [b; params.u_max * ones(obj.udim, 1)];
            elseif size(params.u_max, 1) == obj.udim
                b = [b; params.u_max];
            else
                error("params.u_max should be either a scalar value or an (udim, 1) array.")
            end
        end
        if isfield(params, 'u_min')
            A = [A; -eye(obj.udim), zeros(obj.udim, 1);];
            if size(params.u_min, 1) == 1
                b = [b; -params.u_min * ones(obj.udim, 1)];
            elseif size(params.u_min, 1) == obj.udim
                b = [b; -params.u_min];
            else
                error("params.u_min should be either a scalar value or an (udim, 1) array")
            end
        end        
    else
        % CLF and CBF constraints.
        A = [LgV; -LgB];
        b = [-LfV - params.clf.rate * V;
             LfB + params.cbf.rate * B];                
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
            elseif size(obj.params.u_min, 1) == obj.udim
                b = [b; -params.u_min];
            else
                error("params.u_min should be either a scalar value or an (udim, 1) array")
            end
        end
    end

    %% Cost
    if isfield(params.weight, 'input')
        if size(obj.params.weight.input, 1) == 1 
            weight_input = obj.params.weight.input * eye(obj.udim);
        elseif all(size(obj.params.weight.input) == obj.udim)
            weight_input = obj.params.weight.input;
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
    if with_slack         
        % cost = 0.5 [u' slack] H [u; slack] + f [u; slack]
        H = [weight_input, zeros(obj.udim, 1);
             zeros(1, obj.udim), params.weight.slack];
        f_ = [-weight_input * u_ref; 0];
        [u_slack, ~, exitflag, ~] = quadprog(H, f_, A, b, [], [], [], [], [], options);
        if exitflag == -2
            feas = 0;
            disp("Infeasible QP. CBF constraint is conflicting with input constraints.");
        else
            feas = 1;
        end
        u = u_slack(1:obj.udim);
        slack = u_slack(end);
    else
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
        slack = [];
    end
    comp_time = toc(tstart);
end

function plot_results(t, xs, us, hs, p_o, r_o)

figure
subplot(3,1,1)
plot(t, xs(:,1))
xlabel('t')
ylabel('x [m]')

subplot(3,1,2)
plot(t, xs(:,2))
xlabel('t')
ylabel('y [m]')

subplot(3,1,3)
plot(t, xs(:,3))
xlabel('t')
ylabel('theta [rad]')


figure
plot(t(1:end-1), us)
xlabel('t')
ylabel('u [rad/s]')


lim_min = min(min(xs(:, 1)), min(xs(:, 2)));
lim_max = max(max(xs(:, 1)), max(xs(:, 2)));
lim_min = min([lim_min, p_o(1)-r_o, p_o(2)-r_o]);
lim_max = max([lim_max, p_o(1)+r_o, p_o(2)+r_o]);

figure
plot(xs(:, 1), xs(:, 2));
draw_circle(p_o, r_o);

xlim([lim_min, lim_max]);
ylim([lim_min, lim_max]);
xlabel('x [m]')
ylabel('y [m]')

figure
plot(t(1:end-1), hs)
xlabel('t')
ylabel('cbf h(s)');

end

function h = draw_circle(center,r)
hold on
th = 0:pi/50:2*pi;
xunit = r * cos(th) + center(1);
yunit = r * sin(th) + center(2);
h = plot(xunit, yunit);
hold off

end