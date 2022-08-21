% this file generates fig 4 in 
% 
% https://arxiv.org/abs/2206.11386
%
% "Bi-stochastically normalized graph Laplacian: convergence to
% manifold Laplacian and robustness to outlier noise" 
% by Xiuyuan Cheng, Boris Landa.
%

%%

clear all; rng(2022);

save_fig =0;

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

% SK parameters
boundC = 0.1;
maxite = 50;
discstol = 1e-3;

%% one run

% outlier parameter
p_outlier = .8;% 0.25; %0.8; %outlier fraction p
m = 2000; %ambient dimension
scale_outlier = 0.04; %0.01

epsW = 5e-4; %5e-4;

%%% sample X
tX = sort(rand(Nx,1),'ascend');
dataX = map_to_RD_func(tX);

% add outlier
rhoX = ones(Nx,1);
tmp= rand(Nx,1);
idx_outlier = find( tmp< p_outlier); %find outlier index
% add noise
tmp = zeros(Nx, m);
tmp(:,1:size(dataX,2)) = dataX;
epsm = sqrt(scale_outlier/m);
noise_vector = randn( size( tmp(idx_outlier,:) ))*epsm;
tmp(idx_outlier,:) = tmp(idx_outlier,:) + noise_vector;
dataX = tmp;


%%
% graph laplacian
disXX2 = squareform( pdist(dataX)).^2;

K = exp(- disXX2/(4*epsW));
K = K-diag(diag(K));

%% vis

n_vis = Nx;

ind_vis = randperm(Nx, n_vis);
data_vis = dataX(ind_vis,:);
t_vis = tX(ind_vis);

figure(2),clf;set(gcf,'Position',[100 100 440 386]);
i0 = [1:Nx];
i0(idx_outlier) = [];
scatter3(data_vis(:,1), data_vis(:,2), data_vis(:,3), 40, K(i0(1), ind_vis), 'o', 'filled');
grid on;
colorbar(); view(-56,44);
set(gca,'FontSize',20);
title('data x(1:3)','Interpreter','latex')

% figure(3),clf;
% norm2vec= sqrt(sum(dataX.^2,2));
% plot(tX, norm2vec, '-')
% grid on;


%% tildeW
dK = sum(K,2);
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

B = diag(x)*K*diag(x);
dB = sum(B,2);

figure(9),clf;set(gcf,'Position',[100 100 440 386]);
plot( 1:ite, log10( discs(1:ite)), 'x-', 'LineWidth',2 );
xlabel('iteration', 'Interpreter','latex');
grid on;
title('$\\log_{10}$ r.h.s. discrepancy of SK','Interpreter','latex');
set(gca,'FontSize',20, 'XLim', [1, ite]);

tic,
[v,d]= eigs(diag(dB)-B, diag(dB), maxk, 'sr', 'SubspaceDimension', 50,...
        'MaxIterations', 300, 'Tolerance', 1e-6,'Display', 1);
toc
v = v*sqrt(sum(dB));


[lam2, tmp]=sort(diag(d),'ascend');
v2 = v(:,tmp);

%%

figure(11),clf; set(gcf,'Position',[100 500 1010 386])
subplot(121)
scatter( v1(:,2), v1(:,3), 40, tX);grid on; 
axis equal; 
title('$\hat{L}^{(\rm DM)}_{\rm rw}$', 'Interpreter','latex')
xlabel('$\psi_2$','Interpreter','latex')
ylabel('$\psi_3$','Interpreter','latex')
set(gca,'FontSize',20)
subplot(122), hold on;
plot(tX, v1(:,2:5));
grid on; title('$\psi_j$, $j=2,\cdots,5$','Interpreter','latex')
set(gca,'FontSize',20);

figure(12),clf; set(gcf,'Position',[100 100 1010 386])
subplot(121)
scatter( v2(:,2), v2(:,3), 40, tX);
grid on; 
axis equal;
title('$\hat{L}^{(\rm SK)}_{\rm rw}$', 'Interpreter','latex')
xlabel('$\psi_2$','Interpreter','latex')
ylabel('$\psi_3$','Interpreter','latex')
set(gca,'FontSize',20);
subplot(122), hold on;
plot(tX, v2(:,2:5));
grid on; title('$\psi_j$, $j=2,\cdots,5$','Interpreter','latex')
set(gca,'FontSize',20);

return;

%%
if save_fig
    figure(2), saveas(gcf,'test14_fig2.fig');
    figure(9), saveas(gcf,'test14_fig9.fig');
    figure(11), saveas(gcf,'test14_fig11.fig')
    figure(12), saveas(gcf,'test14_fig12.fig')
end


