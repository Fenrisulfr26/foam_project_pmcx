clc;
clear all;
set(0,'DefaultFigureColormap',feval('jet'));

load('alphabet_binary_10images.mat');
letter = imresize(squeeze(letters(1,:,:)), [100,100], 'nearest');
letter(letter==0) = 2;

load('sensitivity.mat')
smap_re = imresize(sensitivity, [80,80]);

load('data\IRF_061224.mat')
irf = double(IRF(1:1000));
irf = irf-mean(irf(1:100));
irf = irf./max(irf(:));
irf = irf(100:end);

load('data\data_3x3_061224.mat')
[val, ind] = max(irf(:));
c = physconst('LightSpeed');
t0 = 100+ind-round(((2.5+2.5)*1e-2/c)/15e-12);
data = double(flip(data_3x3(:,:,1:833), 3));
data = data(:,:,t0:end);
data = data./max(data(:));


%% downsample
data_time = linspace(15e-12, 15e-12*size(data,3), size(data,3))-15e-12/2;
down_time = linspace(0.1e-9, 10e-9, 101)-0.1e-9/2;

data_down = interp1(data_time(1:667), permute(data(:,:,1:667),[3 1 2]), down_time);
data_down = permute(data_down, [2 3 1]);

irf = irf(1:size(data,3));
irf_down = interp1(data_time(1:667), irf(1:667), down_time);

plot(data_time, irf)
hold on;
plot(down_time, irf_down)

%% check convolution
% cv = conv(irf_down, irf_down, "full");
% plot(cv);
% hold on;
% [val,idx] = max(irf_down);
% plot(circshift(cv,-idx));


%%
global cfg
cfg.nphoton=1e8;

cfg.unitinmm = 1;

cfg.issrcfrom0=1;
cfg.srcdir=[0 0 -1];
mu_s = 18.7; %cm
mu_a = 0.033; %cm
cfg.prop=[0 0 1 1; % background
    mu_a/10 mu_s/10 0 1.37; % homogenous medium
    10 0.01, 0, 1.37; % absorber 
    ]; 

tbins = 101;
cfg.issaveref=1;
cfg.gpuid=1;
cfg.autopilot=1;
cfg.tstart=0;
cfg.tstep=0.1e-9;
cfg.tend=tbins*cfg.tstep;
cfg.maxdetphoton = 1e9;

cfg.debuglevel='P'; % enable the progress bar
cfg.outputtype='flux';

pos = -25:25:25;

cfg.vol = uint8(zeros(310,310,60));
cfg.vol(6:305,6:305,6:55)=1;

smap = zeros([310,310]);
smap(155-39:155+40,155-39:155+40) = smap_re;

transmittance = zeros([3,3,310,310,tbins]);
for src_x=1:3
for src_y=1:3
    cfg.srcpos=[155+pos(src_x) 155+pos(src_y) 60]; 

    fprintf('[%d,%d]\n', src_x, src_y)

    % cfg.vol(6:105,6:105, 30) = letter;

%%
% mcxpreview(cfg);
% title(sprintf('srcpos = [%d,%d]', src_x, src_y));
% saveas(gcf,sprintf('letter_G/srcpos_%d_%d.png',src_x, src_y))
% close all
%%

% Run simulation
[fluence,detpt,vol,seeds]=mcxlab(cfg);
transmittance(src_x,src_y,:,:,:) = squeeze(fluence.dref(:,:,5,:));
% 
% parent = 'scan_src';
% fldr = "letter_G";
% src_fldr = sprintf('srcpos_%d_%d', src_x, src_y);
% if ~exist(sprintf('%s/%s/%s', parent, fldr, src_fldr), 'dir')
%     mkdir(sprintf('%s/%s/%s', parent, fldr, src_fldr));
% end
% % % 
% save(sprintf("%s/%s/%s/abspos_%d_%d_fluence", parent, fldr, src_fldr, N, NN), 'fluence')
% save(sprintf("%s/%s/%s/abspos_%d_%d_cfg", parent, fldr, src_fldr, N, NN), 'cfg')
end
end
%%
% win = 20;
trans = transmittance.*permute(repmat(smap, 1,1,size(transmittance, 5), 3, 3), [4 5 1 2 3]);

sim_time = linspace(cfg.tstep,cfg.tend,cfg.tend/cfg.tstep)-cfg.tstep/2;
data_time = linspace(15e-12, 15e-12*size(data,3), size(data,3))-15e-12/2;
[val,idx] = max(irf_down);
for i=1:3
    for j=1:3
    subplot(3,3,(i-1)*3+j)
    % crop = transmittance(:,:,155-win:155+win,155-win:155+win,:);
    % tmp = crop(i,j,:,:,:);
    tmp = squeeze(transmittance(i,j,:,:,:));
    tmp = tmp.*repmat(smap, 1,1,size(transmittance, 5));
    trace = squeeze(sum(tmp, [1,2]));
    trace = trace./max(squeeze(sum(trans, [3,4])), [], 'all');
    
    % convolve with irf
    tmp2 = conv(trace, irf_down, "full");
    tmp2 = circshift(tmp2,-idx);
    tmp2 = tmp2(1:101);
    trace = tmp2/(max(tmp2(:))/max(trace(:)));

    plot(sim_time, trace)
    hold on;
    plot(data_time, squeeze(data(i,j,:)));
    ylim([0,1]);
    end
end

%% optimize
sim_time = linspace(cfg.tstep,cfg.tend,cfg.tend/cfg.tstep)-cfg.tstep/2;
p0 = [1.87, 0.0033];
opt=optimoptions('lsqcurvefit');
opt.OptimalityTolerance=1E-7;
x = lsqcurvefit(@sim_fn,p0,sim_time,data_down);

%%
global cfg
cfg.nphoton=1e8;
cfg.unitinmm = 1;
cfg.issrcfrom0=1;
cfg.srcdir=[0 0 -1];
mu_s = 18.7; %cm
mu_a = 0.033; %cm
cfg.prop=[0 0 1 1; % background
    mu_a/10 mu_s/10 0 1.37; % homogenous medium
    10 0.01, 0, 1.37; % absorber 
    ]; 
tbins = 101;
cfg.issaveref=1;
cfg.gpuid=1;
cfg.autopilot=1;
cfg.tstart=0;
cfg.tstep=0.1e-9;
cfg.tend=tbins*cfg.tstep;
cfg.maxdetphoton = 1e9;
cfg.debuglevel='P'; % enable the progress bar
cfg.outputtype='flux';
cfg.vol = uint8(zeros(310,310,60));
cfg.vol(6:305,6:305,6:55)=1;

load('sensitivity.mat')
smap_re = imresize(sensitivity, [80,80]);
smap = zeros([310,310]);
smap(155-39:155+40,155-39:155+40) = smap_re;

load('data\IRF_061224.mat')
irf = double(IRF(1:1000));
irf = irf-mean(irf(1:100));
irf = irf./max(irf(:));
irf = irf(100:end);

load('data\data_3x3_061224.mat')
[val, ind] = max(irf(:));
c = physconst('LightSpeed');
t0 = 100+ind-round(((2.5+2.5)*1e-2/c)/15e-12);
data = double(flip(data_3x3(:,:,1:833), 3));
data = data(:,:,t0:end);
data = data./max(data(:));

%downsample
data_time = linspace(15e-12, 15e-12*size(data,3), size(data,3))-15e-12/2;
down_time = linspace(0.1e-9, 10e-9, 101)-0.1e-9/2;
data_down = interp1(data_time(1:667), permute(data(:,:,1:667),[3 1 2]), down_time);
data_down = permute(data_down, [2 3 1]);
irf = irf(1:size(data,3));
irf_down = interp1(data_time(1:667), irf(1:667), down_time);

cfg.smap = smap;
cfg.irf_down = irf_down;


% sim = sim_fn(cfg, 0.09,16.5);

function sim = sim_fn(xdata, p)
global cfg
pos = -25:25:25;
transmittance = zeros([3,3,310,310,101]);
sim = zeros([3,3,101]);
cfg.prop=[0, 0, 1, 1; 0, 0, 0, 1.37; 10, 0.01, 0, 1.37];
cfg.prop(2,1) = p(1);
cfg.prop(2,2) = p(2);
for src_x=1:3
    for src_y=1:3
        cfg.srcpos=[155+pos(src_x) 155+pos(src_y) 60]; 
    
        fprintf('[%d,%d]\n', src_x, src_y)

        fluence=mcxlab(cfg);
        transmittance(src_x,src_y,:,:,:) = squeeze(fluence.dref(:,:,5,:));
    end
end
% multiply by sensitivity
trans = transmittance.*permute(repmat(cfg.smap, 1,1,size(transmittance, 5), 3, 3), [4 5 1 2 3]);
trans_max = max(squeeze(sum(trans, [3,4])), [], 'all');

[val,idx] = max(cfg.irf_down);
for src_x=1:3
    for src_y=1:3
        trace = squeeze(sum(trans(src_x,src_y,:,:,:), [3 4]));
        trace = trace/trans_max;
        % convolve with irf
        tmp2 = conv(trace, cfg.irf_down, "full");
        tmp2 = circshift(tmp2,-idx);
        tmp2 = tmp2(1:101);
        sim(src_x,src_y,:) = tmp2/(max(tmp2(:))/max(trace(:)));
    end
end

end