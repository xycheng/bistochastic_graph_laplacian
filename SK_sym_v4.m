function [x,ite,disps,xs]= SK_sym_v4(A, maxite, boundC, discstol)

% Sinkhorn of u and v

if_verbose = 1;

n = size(A,1);
r = ones(n,1); %target rhs

tol = 1e-6; %terminate algorithm when improvement is less than tol

%%

disps = zeros(1,maxite);
xs = zeros(n,maxite);

%dA = sum(A,2);
%u = sqrt(1./dA); %initialize by inverse of sqrt of row sum
u = ones(n,1);
v = r./(A*u);
x = sqrt(u.*v);

assert( min(x) > boundC);

tic,
for ite = 1:maxite
   
    xs(:,ite) = x;
    rhsdisc =  u.*(A*v) - r;
    disps(ite) = max( abs(rhsdisc));
    
    
    if if_verbose
        fprintf('ite %d: disps = %6.4e\n',ite,disps(ite) );
    end
    
    if disps(ite) < discstol
        break;
    elseif ite >1 && abs(disps(ite)-disps(ite-1))<tol
        break;
    end
    
    u = r./(A*v);
    v = r./(A*u);
    x = sqrt(u.*v);
    
    if sum(x<boundC) > 0
        warning(sprintf('boundC not satisfied in ite %d.\n', ite));
        x( x < boundC) = boundC;
    end
    u=x;
    v=x;
    
end
toc

if ite == maxite
    warning('in SK, maxite achieved');
end


disps = disps(1:ite);
xs = xs(:,1:ite);


return;

