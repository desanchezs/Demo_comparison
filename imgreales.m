clear
%%
imds_no = imageDatastore('brain_tumor_dataset/no/');
imgs_no = readall(imds_no);
%preview any image
imshow(imgs_no{3})
%%
imds_yes = imageDatastore('brain_tumor_dataset/yes/');
imgs_yes = readall(imds_yes);
%preview any image
imshow(imgs_yes{3})
%%
M1_no = zeros(size(imgs_no,1),128,128);
M1_yes = zeros(size(imgs_yes,1),128,128);

for i= 1:size(imgs_no,1)
    M1_no(i,:,:)=tensor(imresize(imgs_no{i}(:,:,1), [128, 128]));
end

for i= 1:size(imgs_yes,1)
    M1_yes(i,:,:)=imresize(imgs_yes{i}(:,:,1), [128, 128]);
end
%% ntrain = 202 observations (78 healthy and 124 sick
s = RandStream('mlfg6331_64'); 
Index_M_yes=randsample(s,size(imgs_yes,1),124,false);
Index_M_no=randsample(s,size(imgs_no,1),78,false);
%%
Index_T_yes=transpose(setdiff([1:size(imgs_yes,1)],Index_M_yes));
Index_T_no =transpose(setdiff([1:size(imgs_no,1)],Index_M_no));
%%
M=[M1_no(Index_M_no,:,:); M1_yes(Index_M_yes,:,:)]; 
yM=[zeros(78,1);ones(124,1)];
T=[M1_no(Index_T_no,:,:); M1_yes(Index_T_yes,:,:)]; 
yT=[zeros(20,1);ones(31,1)];
%%
m=202;
M = tensor(permute(M,[3 2 1]));
A= double(M);
X = randn(m,5); 
b0 = ones(5,1);% n-by-p regular design matrix
%%
A= reshape(A,m,[]);
%%
tic;
disp('rank, n=1000');
[~,beta_rk,g1,dev1] = kruskal_reg(X,M,yM,1,'binomial');
toc;

g1{end}.BIC
%%
n = 128;
ratio = .3;
p = n; q = n;

clear opts
opts.mu = 2^8;
opts.beta = 2^5;
opts.tol = 1E-3;
opts.maxit = 300;
opts.TVnorm = 1;
opts.nonneg = true;

t = cputime;
[U, out] = TVAL3(A,yM,p,q,opts);
t = cputime - t;
%%
I1=double(beta_rk);
I2=U;
%%
T = tensor(permute(T,[3 2 1]));
%%
mtest=51;
T_d= double(T);
T_d= reshape(T_d,mtest,[]);
f1 = T_d*I1(:);
f2 = T_d*I2(:);
ye1= binornd(1, 1./(1+exp(-f1)));
ye2= binornd(1, 1./(1+exp(-f2)));
pi1=1./(1+exp(-f1));
pi2=1./(1+exp(-f2));
%%
[X2,Y2,T2,AUC2] = perfcurve(yT,pi2,1);
AUC2
%%
[X1,Y1,T1,AUC1] = perfcurve(yT,pi1,1);
AUC1
%%
plot(X1,Y1) 
hold on 
plot(X2,Y2) 
hold off
legend('Tensor','V. Total') 
xlabel('FPR'); 
ylabel('TPR'); 
title('ROC Curve') 
hold off

