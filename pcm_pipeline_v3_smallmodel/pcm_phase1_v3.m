%% ================================================================
% PHASE 1: Extract SCIELAB Patch Pairs From TID2013
% ================================================================
clear; clc;

fprintf("\n=== PHASE 1: Extracting S-CIELAB Patches from TID2013 ===\n");

% PARAMETERS
whitePoint = 'D65';
patchDeg = 2;                % DOES NOT MATTER NOW (size normalized to 64×64)
overlap = 0.50;
% rootDir = fullfile('perceptual_color_metrics','TID2013');
rootDir = fullfile('TID2013');
% rootDir = fullfile('perceptual_color_metrics','test');

% Find I01..I25
refFolders = dir(fullfile(rootDir, 'I*'));
refFolders = refFolders([refFolders.isdir]);
fprintf("Found %d reference image folders.\n", numel(refFolders));

allSamples = {};
allYvals = [];

%% MAIN LOOP
for f = 1:numel(refFolders)

    folderName = refFolders(f).name;
    folderPath = fullfile(refFolders(f).folder, folderName);

    fprintf("\n-- Folder: %s\n", folderName);

    refPath = fullfile(folderPath, sprintf('%s.bmp', folderName));
    
    if ~isfile(refPath)
        warning("Missing reference: %s", refPath);
        continue;
    end

    % --- Load reference XYZ ---
    refXYZ = readSceneXYZ(refPath, whitePoint);

    % Find distorted images
    distFiles = dir(fullfile(folderPath, '*.bmp'));
    distFiles = distFiles(~strcmp({distFiles.name}, sprintf('%s.bmp',folderName)));
    distFiles = distFiles(1:10:end);

    fprintf("Found %d distorted images.\n", numel(distFiles));

    for k = 1:numel(distFiles)

        distPath = fullfile(distFiles(k).folder, distFiles(k).name);

        fprintf("\nExtracting Pairs from... %s", distPath);

        % Load distorted image
        distXYZ = readSceneXYZ(distPath, whitePoint);

        % Extract 64×64 paired patches
        batch = xyzToPatchPairs(refXYZ, distXYZ, patchDeg, overlap);

        % Store
        for b = 1:numel(batch)
            X = batch{b}{1};
            Y = batch{b}{2};
            allSamples{end+1,1} = {X, Y};
            allYvals = [allYvals; Y(:)];
        end
    end
end

fprintf("\nTotal extracted samples: %d\n", numel(allSamples));

%% NORMALIZATION
meanY = mean(allYvals(:));
stdY = std(allYvals(:));

fprintf("\nNormalization: mean = %.4f, std = %.4f\n", meanY, stdY);

for i = 1:numel(allSamples)
    Y = allSamples{i}{2};
    allSamples{i}{2} = (Y - meanY) ./ stdY;
end

%% SAVE DATASET
save("SCIELAB_dataset.mat","allSamples","meanY","stdY","-v7.3");
fprintf("\nSaved: SCIELAB_dataset.mat\n");
