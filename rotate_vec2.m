function v= rotate_vec2( v, v_true)

[n, k] = size(v_true); 
assert(k==2);
assert(size(v,1)==n);
assert(size(v,2)==2);


%% align sign 
for j=1:2
    v(:,j) =v(:,j)* sign( v(:,j)'*v_true(:,j));
end


%% rotate eigenvectors to align, due to multiplicity 2

rot = v\v_true;
[uu,ss,vv]=svd(rot);
rot = uu*vv';

v = v*rot;

%% allow a scalor
for j=1:2
    beta = norm( v_true(:,j))/norm(v(:,j));
    v(:,j) = v(:,j) *beta;
end

end