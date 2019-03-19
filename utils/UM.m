function [imgTanhF]=IntEdgUMF()
% Paper: S.C.F. Lin, C.Y. Wong, G. Jiang, M.A. Rahman, T.R. Ren, Ngaiming Kwok,
% Haiyan Shib, Ying-Hao Yuc, Tonghai Wu, "Intensity and edge based adaptive
% unsharp masking filter for colorimage enhancement," Optik 127 (2016)
% 407?414


clc; clear all; close all; dbstop if error;
%{
img.h=460; img.w=700; img.L=255; img.L_=linspace(0,img.L,img.L+1);
job=0; fig=[];
[fn,pn,bn]=uigetfile('test\\*.png');
pn
fn

if bn>0,
  img.PNG=imread([pn fn]);
  [img,img.RGB]=MyImResize(img,img.PNG);
  fig(end+1)=MyFigure(); MyImshow(fig(end),img.RGB,fn,job); job=job+1;
  [imgTanhF,ovr,k]=MyTanhF(img,img.RGB);
  fig(end+1)=MyFigure(); MyImshow(fig(end),imgTanhF,fn,job);
end;
%}


path = "dataset/breakhis/";
results = "um_dataset/";
img.h=460; img.w=700; img.L=255; img.L_=linspace(0,img.L,img.L+1); 

files = dir(path);
for file = files'
    dot = strcmp(file.name, ".");
    dotdot = strcmp(file.name, "..");
    condition = dot | dotdot;
    if ~(condition)
        path_i = strcat(path,file.name);
        path_r = strcat(results,file.name);
        img.PNG = imread(path_i);
        [img,img.RGB]=MyImResize(img,img.PNG);
        [imgTanhF,ovr,k]=MyTanhF(img,img.PNG);
        imwrite(imgTanhF, path_r);
    end
end

function [kmg,ovr,k]=MyTanhF(img,jmg)
H=[-1 -1 -1; -1 8 -1; -1 -1 -1];
jmg=MyStretch(img,jmg);
HSI=rgb2hsi(jmg); GRY=HSI(:,:,3);
MKF_=imfilter(GRY,H,'same'); MKF=zeros(size(MKF_));
MKF(2:end-1,2:end-1)=MKF_(2:end-1,2:end-1);
kL=0; kH=2; rng=kH-kL; rho=0.5*(sqrt(5)-1); tol=0.01;
k(1)=kL+(1-rho)*rng; k(2)=kL+rho*rng;
[ENH(:,:,1),ent(1),ovr(1)]=MyGolden(k(1),GRY,MKF,img);
[ENH(:,:,2),ent(2),ovr(2)]=MyGolden(k(2),GRY,MKF,img);
while rng>tol,
  k_=k; ovr_=ovr;
  if ent(1)>ent(2),
    kH=k(2); rng=kH-kL; k(1)=kL+(1-rho)*rng; k(2)=kL+rho*rng; ent(2)=ent(1);
    [ENH,ent(1),ovr(1)]=MyGolden(k(1),GRY,MKF,img);
  else
    kL=k(1); rng=kH-kL; k(1)=kL+(1-rho)*rng; k(2)=kL+rho*rng; ent(1)=ent(2);
    [ENH,ent(2),ovr(2)]=MyGolden(k(2),GRY,MKF,img);
  end;
end;
ovr=mean(ovr_); k=mean(k_);
HSI(:,:,3)=ENH; kmg=hsi2rgb(HSI); kmg=uint8(kmg*img.L);
function [ENH,ent,ovr]=MyGolden(k,GRY,MKF,img)
TAH1=0.5*(1+tanh(3-12*abs(GRY-0.5)));
TAH2=0.5*(1+tanh(3-(6*abs(MKF)-0.5)));
TAH=TAH1.*TAH2;
ENH=GRY+k*TAH.*MKF;
[ENH,ovr]=MyRestore(ENH,GRY,img);
ent=entropy(ENH(2:end-1,2:end-1))*(1-ovr);
function [ENH,ovr]=MyRestore(ENH,GRY,img)
z0=find(ENH<0); ENH(z0)=GRY(z0);
z1=find(ENH>1); ENH(z1)=GRY(z1);
ovr=length(z0)+length(z1); ovr=ovr/img.N;
function [RGB]=MyStretch(img,RGB)
RGB=double(RGB)/img.L;
for k=1:size(RGB,3),
  mi=min(min(RGB(:,:,k))); RGB(:,:,k)=RGB(:,:,k)-mi;
  mx=max(max(RGB(:,:,k))); RGB(:,:,k)=RGB(:,:,k)/mx;
end;
RGB=uint8(RGB*img.L);
function [fig]=MyFigure()
mon=get(0,'MonitorPositions');
x=(rand*0.1+0.1)*mon(3);
y=(rand*0.1+0.2)*mon(4);
fig=figure('units','pixel','position',[x y 600 400]);
function []=MyImshow(fig,jmg,fn,job)
jmg=jmg(2:end,2:end,:);% remove boundary
figure(fig); imshow(uint8(jmg));
set(gca,'position',[0 0 1 1]);
set(gcf,'name',fn); drawnow;
function [img,jmg]=MyImResize(img,jmg)
[v,u,w]=size(jmg);
if u>v,% landscape
  k=[img.h img.w];
else% portrait
  k=[img.w img.h];
end;
jmg=imresize(jmg,k,'bicubic');% resize
[img.V,img.U,img.N]=size(jmg);% image size 
img.N=img.V*img.U;
function hsi = rgb2hsi(rgb)
%RGB2HSI Converts an RGB image to HSI.
% Extract the individual component images.
rgb = im2double(rgb);
r = rgb(:, :, 1);
g = rgb(:, :, 2);
b = rgb(:, :, 3);
% Implement the conversion equations.
num = 0.5*((r - g) + (r - b));
den = sqrt((r - g).^2 + (r - b).*(g - b));
theta = acos(num./(den + eps));
H = theta;
H(b > g) = 2*pi - H(b > g);
H = H/(2*pi);
num = min(min(r, g), b);
den = r + g + b;
den(den == 0) = eps;
S = 1 - 3.* num./den;
H(S == 0) = 0;
I = (r + g + b)/3;
% Combine all three results into an hsi image.
hsi = cat(3, H, S, I);
function rgb = hsi2rgb(hsi)
%HSI2RGB Converts an HSI image to RGB.
% Extract the individual HSI component images.
H = hsi(:, :, 1) * 2 * pi;
S = hsi(:, :, 2);
I = hsi(:, :, 3);
% Implement the conversion equations.
R = zeros(size(hsi, 1), size(hsi, 2));
G = zeros(size(hsi, 1), size(hsi, 2));
B = zeros(size(hsi, 1), size(hsi, 2));
% RG sector (0 <= H < 2*pi/3).
idx = find( (0 <= H) & (H < 2*pi/3));
B(idx) = I(idx) .* (1 - S(idx));
R(idx) = I(idx) .* (1 + S(idx) .* cos(H(idx)) ./ ...
                                          cos(pi/3 - H(idx)));
G(idx) = 3*I(idx) - (R(idx) + B(idx));
% BG sector (2*pi/3 <= H < 4*pi/3).
idx = find( (2*pi/3 <= H) & (H < 4*pi/3) );
R(idx) = I(idx) .* (1 - S(idx));
G(idx) = I(idx) .* (1 + S(idx) .* cos(H(idx) - 2*pi/3) ./ ...
                    cos(pi - H(idx)));
B(idx) = 3*I(idx) - (R(idx) + G(idx));
% BR sector.
idx = find( (4*pi/3 <= H) & (H <= 2*pi));
G(idx) = I(idx) .* (1 - S(idx));
B(idx) = I(idx) .* (1 + S(idx) .* cos(H(idx) - 4*pi/3) ./ ...
                                           cos(5*pi/3 - H(idx)));
R(idx) = 3*I(idx) - (G(idx) + B(idx));
% Combine all three results into an RGB image.  Clip to [0, 1] to
% compensate for floating-point arithmetic rounding effects.
rgb = cat(3, R, G, B);
rgb = max(min(rgb, 1), 0);