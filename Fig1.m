%% ---- Select specific reference and distortion type ----
ieInit;

tidRoot = 'D:\TID2013';
refDir  = fullfile(tidRoot, 'reference_images');
distDir = fullfile(tidRoot, 'distorted_images');

r0 = 1;   % choose reference i03.bmp (you can change)
refFiles = dir(fullfile(refDir, '*.bmp'));
refName  = refFiles(r0).name;
refID    = refName(1:3);
refPath  = fullfile(refDir, refName);

fprintf('Selected reference = %s\n', refName);

%% ---- Distortion type 01, levels 1 and 5 ----
typeID = '01';
levels = [1, 5];   % LV1 and LV5

distPaths = cell(1,2);
for k = 1:2
    distName = sprintf('%s_%s_%d.bmp', refID, typeID, levels(k));
    distPaths{k} = fullfile(distDir, distName);
    fprintf('Selected distorted lvl%d = %s\n', levels(k), distName);
end

%% ---- Compute S-CIELAB for LV1 & LV5 ----
dispCal = 'crt.mat';
vDist   = 0.3;

[e1, sRef, sTest1] = scielabRGB(refPath, distPaths{1}, dispCal, vDist);
[e5, ~,     sTest5] = scielabRGB(refPath, distPaths{2}, dispCal, vDist);

%% ---- Resize XYZ size match (if needed) ----
xyz1 = sceneGet(sTest1,'xyz');
xyz5 = sceneGet(sTest5,'xyz');

[er, ec] = size(e1);
xyz1 = imresize(xyz1, [er ec]);
xyz5 = imresize(xyz5, [er ec]);

%% ---- Plot 2x2 Figure ----
figure('Position',[100 100 1200 900]);

%% ---- Custom subplot positions (reduce middle gaps) ----
pos_ref  = [0.07 0.52 0.40 0.40];   % left top
pos_noi1 = [0.53 0.52 0.40 0.40];   % right top
pos_lv1  = [0.07 0.08 0.40 0.38];   % left bottom
pos_lv5  = [0.53 0.08 0.40 0.38];   % right bottom

%% ---- Plot Reference ----
axes('Position', pos_ref);
imshow(sceneGet(sRef,'rgb'));
title('Reference', 'FontSize', 16);

%% ---- Plot Distorted (LV1) ----
axes('Position', pos_noi1);
imshow(sceneGet(sTest1,'rgb'));
title('Gaussian Noise Level 1', 'FontSize', 16);

%% ---- Plot Heatmap LV1 ----
axes('Position', pos_lv1);
imagesc(e1); axis image off;
c = colorbar; c.FontSize = 12;
title('\DeltaE (LV1)', 'FontSize', 16);

%% ---- Plot Heatmap LV5 ----
axes('Position', pos_lv5);
imagesc(e5); axis image off;
c = colorbar; c.FontSize = 12;
title('\DeltaE (LV5)', 'FontSize', 16);