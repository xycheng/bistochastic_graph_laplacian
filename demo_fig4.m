% this file generates fig 4 and Fig A.1 in 
% 
% https://arxiv.org/abs/2206.11386
%
% "Bi-stochastically normalized graph Laplacian: convergence to
% manifold Laplacian and robustness to outlier noise" 
% by Xiuyuan Cheng, Boris Landa.
%

%%
clear all; rng(2022);

noise_type = '1'; %set '1' to reproduce Fig 4, set '2' to reproduce Fig A.1

if_plot = 1;

%%
dM =1;

omegaM = 2; 

map_to_RD_func = @(t) 1/(sqrt(5)*2*pi)*[...
                       cos(2*pi * t), ...
                       sin(2*pi * t), ...
                       2/omegaM*cos(2*pi * omegaM*t), ...
                       2/omegaM*sin(2*pi * omegaM*t)];


%% sample X and create kernel matrix
Nx = 1000; 

m = 2000; %ambient dimension
scale_outlier = 0.01;

epsW = 0.0005; 

% outlier parameter
p_out_max = 0.95; %max outlier fraction p
p_out_min = .05;


% SK parameters 
maxite = 50;
discstol = 1e-3;
boundC = 1e-8;
    % this is the placeholder of boundC, which does not inforce the
    % projection step in the SK algorithm. observe that the final empirical
    % factor observe some posterior boundC.

%% one run

tX = sort(rand(Nx,1),'ascend');
dataX = map_to_RD_func(tX);
dataX_c = dataX;

% add noise
tmp = zeros(Nx, m);
tmp(:,1:size(dataX,2)) = dataX;

switch noise_type
    case '1'
        % heteroskedastic bi and zi
        meanb = p_out_min+ (p_out_max-p_out_min)* ( mod(1-tX+rand(1),1) );
        bX = (rand(Nx,1) < meanb);
        idx_outlier = find( bX==1); %find outlier index
        
        rho_per_sample1 = 10.^( (1-((1+sin(tX(idx_outlier)*2*pi))*0.5).^2 )*1 );
        rho_per_sample2 = rand( numel(idx_outlier), 1)*3;
        
        rho_per_sample = 0.9*rho_per_sample1+0.1*rho_per_sample2;
    case '2'
        % i.i.d bi and zi
        p_outlier = p_out_max;
        bX = (rand(Nx,1) < p_outlier);
        idx_outlier = find( bX==1); %find outlier index
        
        rho_per_sample1 = 10.^( (1-((1+sin(tX(idx_outlier)*2*pi))*0.5).^2 )*1 );
        rho_per_sample2 = rand( numel(idx_outlier), 1)*3;
        
        rho_per_sample = rho_per_sample2;
end

epsm_per_sample =  sqrt(rho_per_sample*scale_outlier/m );
noise_vector = randn( size( tmp(idx_outlier,:) ));
noise_vector = bsxfun(@times, epsm_per_sample, noise_vector);

tmp(idx_outlier,:) = tmp(idx_outlier,:) + noise_vector;
dataX = tmp;

%% graph laplacian
disXX2 = squareform( pdist(dataX)).^2;

K = exp(- disXX2/(4*epsW));
K = K-diag(diag(K));

if if_plot
    figure(4),clf;
    plot( tX(idx_outlier), rho_per_sample, 'x');
    xlabel('tX'); ylabel('magnitude of noise')
    grid on;
    set(gca,'FontSize',20);
    
    % plot data
    n_vis = Nx;
    ind_vis = randperm(Nx, n_vis);
    data_vis = dataX(ind_vis,:);
    t_vis = tX(ind_vis);
    
    figure(2),clf; %set(gcf,'Position',[100 100 440 386]);
    i0 = [1:Nx];
    i0(idx_outlier) = [];
    scatter3(data_vis(:,1), data_vis(:,2), data_vis(:,3), 40, K(i0(1), ind_vis), 'o', 'filled');
    grid on;
    colorbar(); view(-56,44);
    set(gca,'FontSize',20);
    title('data x(1:3)','Interpreter','latex')
end


%% tildeW
dK = sum(K,2);

fprintf('min dK = %6.4e.\n', min(dK) )

if min(dK) < 1e-6
    warning(sprintf('min dK too small: %6.4e.\n', min(dK) ));
end

tildeW = K./(sqrt(dK)*sqrt(dK)');
dW = sum(tildeW,2);

%% eigen of L_rw
tic,
maxk =  10;
[v,d]= eigs(diag(dW)-tildeW, diag(dW), maxk, 'sr', 'SubspaceDimension', 50,...
    'MaxIterations', 300, 'Tolerance', 1e-6, 'Display', 1);

toc
v = v*sqrt(sum(dW));

[lam1, tmp]=sort(diag(d),'ascend');
v1 = v(:,tmp);

%% SK
[x,ite,discs,xs]= SK_sym_v4(K, maxite, boundC, discstol);
fprintf('after SK terminates, min eta= %6.4e. This is posterior boundC.\n', min(x) )

B = diag(x)*K*diag(x);
B = (B+B')/2;
dB = sum(B,2);

if if_plot
    figure(9),clf; 
    plot( 1:ite, log10( discs(1:ite)), 'x-', 'LineWidth',2 );
    xlabel('iteration', 'Interpreter','latex');
    grid on;
    title('$\\log_{10}$ r.h.s. discrepancy of SK','Interpreter','latex');
    set(gca,'FontSize',20, 'XLim', [1, ite]);
    
    figure(8), clf;  hold on;
    id=1:Nx; id(idx_outlier)=[];
    plot( tX(id), x(id), 'o-')
    plot(tX(idx_outlier), x(idx_outlier), 'x')
    %axis([0 1 -0.1 max(x(id))*10]); 
          %zoom in to look at the factor on inliers
    plot(tX, ones(size(tX))*boundC, '--')
    grid on;
    legend('inlier', 'outlier','boundC')
    set(gca,'FontSize',20);
end

tic,
[v,d]= eigs(diag(dB)-B, diag(dB), maxk, 'sr', 'SubspaceDimension', 50,...
    'MaxIterations', 300, 'Tolerance', 1e-6,'Display', 1);
toc
v = v*sqrt(sum(dB));

[lam2, tmp]=sort(diag(d),'ascend');
v2 = v(:,tmp);


%% align to limiting harmonics
v1_align = zeros(Nx,4);
v2_align = zeros(Nx,4);
v_true = zeros(Nx,4);

v_true(:,1:2) = [cos(tX*2*pi), sin(tX*2*pi)];
v1_align(:,1:2) = rotate_vec2( v1(:,2:3), v_true(:,1:2));
v2_align(:,1:2) = rotate_vec2( v2(:,2:3), v_true(:,1:2));

v_true(:,3:4) = [cos(2*tX*2*pi), sin(2*tX*2*pi)];
v1_align(:,3:4) = rotate_vec2( v1(:,4:5), v_true(:,3:4));
v2_align(:,3:4) = rotate_vec2( v2(:,4:5), v_true(:,3:4));

err1 = sum( reshape(sum((v1_align - v_true).^2,1)/Nx, [2,2] ),1)
err2 = sum( reshape( sum((v2_align - v_true).^2,1)/Nx, [2,2]), 1)

%%
if if_plot
    
    figure(11),clf; 
    subplot(121),
    scatter( v1_align(:,1), v1_align(:,2), 40, tX);grid on;
    a = max(max(  abs(v1_align(:,1:2)) ) )*1.1;
    axis([-a a -a a])
    title('$\hat{L}^{(\rm DM)}_{\rm rw}$', 'Interpreter','latex')
    xlabel('$\psi_2$','Interpreter','latex')
    ylabel('$\psi_3$','Interpreter','latex')
    set(gca,'FontSize',20)
    subplot(122), hold on;
    plot(tX, v1_align);
    grid on; title('$\psi_j$, $j=2,\cdots,5$','Interpreter','latex')
    set(gca,'FontSize',20);
    
    figure(12),clf; 
    subplot(121)
    scatter( v2_align(:,1), v2_align(:,2), 40, tX);grid on;
    a = max(max(  abs(v2_align(:,1:2)) ) )*1.1;
    axis([-a a -a a])
    title('$\hat{L}^{(\rm SK)}_{\rm rw}$', 'Interpreter','latex')
    xlabel('$\psi_2$','Interpreter','latex')
    ylabel('$\psi_3$','Interpreter','latex')
    set(gca,'FontSize',20);
    subplot(122), hold on;
    plot(tX, v2_align);
    axis([0 1 -1.5 1.5])
    grid on; title('$\psi_j$, $j=2,\cdots,5$','Interpreter','latex')
    set(gca,'FontSize',20);
    
    drawnow();
end


return;


