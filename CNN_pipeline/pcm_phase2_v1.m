%% ================================================================
% PHASE 2: Training UNET for S-CIELAB
% ================================================================

clear; clc;

fprintf("\n=== PHASE 2: Training UNET for S-CIELAB ===\n");

%% LOAD DATASET
load("SCIELAB_dataset.mat","allSamples","meanY","stdY");
N = numel(allSamples);
fprintf("Loaded %d samples.\n", N);

%% DATASTORE
flatDs = arrayDatastore(allSamples,"OutputType","same");
ds = transform(flatDs, @(c) c{1});  % unwrap {{X,Y}} → {X,Y}

%% TRAIN/VAL SPLIT
idx = randperm(N);
Ntrain = round(0.8 * N);
dsTrain = subset(ds, idx(1:Ntrain));
dsVal   = subset(ds, idx(Ntrain+1:end));

fprintf("Training: %d\n", Ntrain);
fprintf("Validation: %d\n", N-Ntrain);

%% NETWORK INPUT
inputSize = [64 64 6];

%% UNET ARCHITECTURE
layers = [
    imageInputLayer(inputSize,"Normalization","none","Name","input")

    % ------------ ENCODER 1 ------------
    convolution2dLayer(3,32,"Padding","same","Name","conv1_1")
    reluLayer("Name","relu1_1")
    convolution2dLayer(3,32,"Padding","same","Name","conv1_2")
    reluLayer("Name","relu1_2")
    maxPooling2dLayer(2,"Stride",2,"Name","pool1")

    % ------------ ENCODER 2 ------------
    convolution2dLayer(3,64,"Padding","same","Name","conv2_1")
    reluLayer("Name","relu2_1")
    convolution2dLayer(3,64,"Padding","same","Name","conv2_2")
    reluLayer("Name","relu2_2")
    maxPooling2dLayer(2,"Stride",2,"Name","pool2")

    % ------------ BOTTLENECK ------------
    convolution2dLayer(3,128,"Padding","same","Name","conv3_1")
    reluLayer("Name","relu3_1")
    convolution2dLayer(3,128,"Padding","same","Name","conv3_2")
    reluLayer("Name","relu3_2")

    % ------------ DECODER 1 ------------
    transposedConv2dLayer(2,64,"Stride",2,"Name","up1")
    depthConcatenationLayer(2,"Name","concat1")
    convolution2dLayer(3,64,"Padding","same","Name","conv4_1")
    reluLayer("Name","relu4_1")
    convolution2dLayer(3,64,"Padding","same","Name","conv4_2")
    reluLayer("Name","relu4_2")

    % ------------ DECODER 2 ------------
    transposedConv2dLayer(2,32,"Stride",2,"Name","up2")
    depthConcatenationLayer(2,"Name","concat2")
    convolution2dLayer(3,32,"Padding","same","Name","conv5_1")
    reluLayer("Name","relu5_1")
    convolution2dLayer(3,32,"Padding","same","Name","conv5_2")
    reluLayer("Name","relu5_2")

    convolution2dLayer(1,1,"Padding","same","Name","conv_out")
    regressionLayer
];

%% SKIP CONNECTIONS
lgraph = layerGraph(layers);
lgraph = connectLayers(lgraph,"relu2_2","concat1/in2");
lgraph = connectLayers(lgraph,"relu1_2","concat2/in2");

%% TRAINING OPTIONS
options = trainingOptions("adam", ...
    "InitialLearnRate", 3e-4, ...
    "MiniBatchSize", 32, ...
    "MaxEpochs", 8, ...
    "ValidationData", dsVal, ...
    "ValidationFrequency", 200, ...
    "Shuffle", "every-epoch", ...
    "LearnRateSchedule", "piecewise", ...
    "LearnRateDropFactor", 0.5, ...
    "LearnRateDropPeriod", 12, ...
    "Plots", "training-progress", ...
    "Verbose", true);

%% TRAIN
fprintf("\nStarting training...\n");
net = trainNetwork(dsTrain, lgraph, options);

%% SAVE
save("SCIELAB_UNET_model.mat","net","meanY","stdY","-v7.3");
fprintf("\nSaved model: SCIELAB_UNET_model.mat\n");

%% EVALUATE MODEL

metrics = evaluate_UNET_scielab(net, dsVal, meanY, stdY);
