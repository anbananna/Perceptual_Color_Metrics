%% ========================================================================
%  tid2013_scielab_mlp.m
%
%  TID2013 + S-CIELAB + MLP (mean DeltaE prediction)
%
%  - Uses TID2013 reference and distorted images
%  - Computes S-CIELAB DeltaE maps via scielabRGB
%  - Cuts 2x2 degree non-overlapping patches
%  - For each patch, extracts 18-D XYZ statistics from ref + distorted
%  - Trains an MLP (mlpNet) to predict mean DeltaE per patch
% ========================================================================

%% 0. Setup ISETCam and paths
ieInit;  % ISETCam init

% ---- CHANGE THIS PATH TO YOUR TID2013 FOLDER ----
tidRoot = 'D:\TID2013';   % TODO: change this

refDir  = fullfile(tidRoot, 'reference_images');
distDir = fullfile(tidRoot, 'distorted_images');

if ~exist(refDir,'dir') || ~exist(distDir,'dir')
    error('Check tidRoot. refDir or distDir does not exist.');
end

%% 1. Read all reference images (BMP only, TID2013)
refFiles = dir(fullfile(refDir, '*.bmp'));
nRefTotal = numel(refFiles);
fprintf('Total reference images found: %d\n', nRefTotal);

if nRefTotal == 0
    error('No reference BMP images found in %s', refDir);
end

% For safety, start with a small subset
maxRefsToUse   = min(25, nRefTotal);   % you can increase later
maxDistsPerRef = 120;                  % number of distorted images per ref

% Distortion directory: each distorted file is like i01_01_1.bmp etc.
distFilesAll = dir(fullfile(distDir, '*.bmp'));
nDistTotal   = numel(distFilesAll);
fprintf('Total distorted images found: %d\n', nDistTotal);

%% 2. Global parameters
hfovDeg  = 20;   % horizontal field of view in degrees (used for patch size)
patchDeg = 2;    % patch covers 2x2 degrees

dispCal = 'crt.mat';  % display calibration for scielabRGB
vDist   = 0.3;        % viewing distance in meters

%% 3. Data containers
allFeatures = {};
allLabels   = [];
sampleCount = 0;

%% ======================= MAIN LOOP OVER REFERENCES ======================
for r = 1:maxRefsToUse
    refName = refFiles(r).name;   % e.g., 'i01.bmp'
    refID   = refName(1:3);       % e.g., 'i01'
    refPath = fullfile(refDir, refName);

    fprintf('\n=== [%d/%d] Reference: %s ===\n', r, maxRefsToUse, refName);

    % Find distorted images belonging to this reference
    distPattern = sprintf('%s_*.bmp', refID);   % e.g., 'i01_*.bmp'
    distFiles   = dir(fullfile(distDir, distPattern));
    if isempty(distFiles)
        warning('No distorted images found for %s. Skipping.', refID);
        continue;
    end

    nDists = numel(distFiles);
    nUse   = min(maxDistsPerRef, nDists);

    fprintf('  Distorted images found: %d, using first %d\n', nDists, nUse);

    %% Loop over a subset of distorted images
    for d = 1:nUse
        distName = distFiles(d).name;
        distPath = fullfile(distDir, distName);
        fprintf('    Distorted: %s\n', distName);

        try
            % eImage: DeltaE map; sRef/sTest: scenes
            [eImage, sRef, sTest] = scielabRGB(refPath, distPath, dispCal, vDist);
        catch ME
            warning('    scielabRGB failed: %s. Skipping this distorted.', ME.message);
            continue;
        end

        % Get XYZ from scenes
        xyzRef = sceneGet(sRef,  'xyz');
        xyzTst = sceneGet(sTest, 'xyz');

        % Make sure XYZ and DeltaE map have the same spatial size.
        % If not, resize XYZ to match eImage.
        [erows, ecols] = size(eImage);
        [rrows, rcols, ~] = size(xyzRef);

        if erows ~= rrows || ecols ~= rcols
            % Resize XYZ to eImage size (no skipping)
            xyzRef = imresize(xyzRef, [erows ecols]);
            xyzTst = imresize(xyzTst, [erows ecols]);
            fprintf('      Resized XYZ to match DeltaE map: [%d x %d]\n', erows, ecols);
        end

        % Set HFOV just for computing degrees per pixel
        sRef = sceneSet(sRef, 'hfov', hfovDeg);
        hfov = sceneGet(sRef, 'hfov');   % should be hfovDeg
        degPerPixel = hfov / ecols;
        patchPix    = max(1, round(patchDeg / degPerPixel));

        % Number of non-overlapping patches
        nRow = floor(erows / patchPix);
        nCol = floor(ecols / patchPix);
        fprintf('      Patches: %d x %d = %d\n', nRow, nCol, nRow * nCol);

        % Loop over patches
        for rr = 1:nRow
            for cc = 1:nCol

                r1 = (rr-1)*patchPix + 1; r2 = r1 + patchPix - 1;
                c1 = (cc-1)*patchPix + 1; c2 = c1 + patchPix - 1;

                xyzRefPatch = xyzRef(r1:r2, c1:c2, :);
                xyzTstPatch = xyzTst(r1:r2, c1:c2, :);
                dEPatch     = eImage(r1:r2, c1:c2);

                meanDE = mean(dEPatch(:));   % label
                feat   = extractXYZPatchFeatures18(xyzRefPatch, xyzTstPatch);  % 18-D

                sampleCount = sampleCount + 1;
                allFeatures{sampleCount,1} = feat;
                allLabels(sampleCount,1)   = meanDE;
            end
        end

        % Clear large variables for safety
        clear xyzRef xyzTst eImage sRef sTest;

    end % end distorted loop
end % end reference loop

fprintf('\nTotal patch samples collected: %d\n', sampleCount);

%% 4. Build feature matrix X and label vector Y
X = cell2mat(allFeatures);    % [N x D]
Y = allLabels(:);             % [N x 1]

[N, D] = size(X);
fprintf('Feature matrix size: %d x %d\n', N, D);

if N < 10
    warning('Very few samples. Consider using more references or distortions.');
end

%% 5. Split into training and test sets
testRatio = 0.2;
Ntest  = max(1, round(N * testRatio));
idx    = randperm(N);

idxTest  = idx(1:Ntest);
idxTrain = idx(Ntest+1:end);

Xtrain = X(idxTrain,:);
Ytrain = Y(idxTrain,:);
Xtest  = X(idxTest,:);
Ytest  = Y(idxTest,:);

fprintf('Training samples: %d, Test samples: %d\n', numel(Ytrain), numel(Ytest));

%% 6. Feature standardization (zero-mean, unit-variance per feature)
[XtrainN, ps] = mapstd(Xtrain');   % mapstd works on rows
XtrainN = XtrainN';                % back to [N x D]

XtestN  = mapstd('apply', Xtest', ps)';
fprintf('Standardized features.\n');

%% 7. Train MLP (mlpNet)
inputDim = size(XtrainN,2);
fprintf('Input feature dimension: %d\n', inputDim);
rng(0);
hiddenSizes = [64 32];  % you can tune this
mlpNet = fitnet(hiddenSizes);

mlpNet.divideParam.trainRatio = 0.9;
mlpNet.divideParam.valRatio   = 0.1;
mlpNet.divideParam.testRatio  = 0;
mlpNet.trainParam.epochs      = 300;
mlpNet.trainParam.showWindow  = true;

fprintf('Start training mlpNet...\n');
mlpNet = train(mlpNet, XtrainN', Ytrain');   % inputs: [D x N], targets: [1 x N]

%% 8. Evaluate on training and test sets
YpredTrain = mlpNet(XtrainN')';
R_train    = corr(YpredTrain, Ytrain);
fprintf('Training set R = %.4f\n', R_train);

Ypred = mlpNet(XtestN')';
err   = Ypred - Ytest;

MAE  = mean(abs(err));
RMSE = sqrt(mean(err.^2));
R    = corr(Ypred, Ytest);

fprintf('\n===== Test set performance =====\n');
fprintf('MAE  = %.4f\n', MAE);
fprintf('RMSE = %.4f\n', RMSE);
fprintf('R    = %.4f\n', R);

figure;
scatter(Ytest, Ypred, 20, 'filled');
grid on;
xlabel('True mean S-CIELAB \DeltaE');
ylabel('Predicted \DeltaE (MLP)');
title(sprintf('Test set: R = %.3f, RMSE = %.3f', R, RMSE));
save('D:\TID2013\scielab_mlp_model1.mat', ...
     'mlpNet', ...      % trained MLP
     'ps', ...          % parameters
     'R', 'RMSE', 'MAE');  % final result

fprintf('? Trained MLP model saved as scielab_mlp_model.mat\n');

%% ======================= Helper function ===============================
function feat = extractXYZPatchFeatures18(xRef, xTest)
% xRef, xTest: [h x w x 3] XYZ patches (reference and distorted)
% Features (18-D):
%   ref:  meanXYZ + stdXYZ -> 6
%   test: meanXYZ + stdXYZ -> 6
%   diff: (meanT - meanR) + (stdT - stdR) -> 6
%   total = 18

    R = reshape(xRef,  [], 3);
    T = reshape(xTest, [], 3);

    mR = mean(R,1); sR = std(R,0,1);
    mT = mean(T,1); sT = std(T,0,1);

    dm = mT - mR;
    ds = sT - sR;

    feat = [mR, sR, mT, sT, dm, ds];   % 1 x 18
end