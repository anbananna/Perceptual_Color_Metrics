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
    imageInputLayer([64 64 6],"Normalization","none","Name","input")

    % --- Initial Feature Extraction ---
    convolution2dLayer(5,64,"Padding","same","Name","conv_in")
    batchNormalizationLayer("Name","bn_in")
    reluLayer("Name","relu_in")

    % --- Residual Block 1 (dilation 1) ---
    convolution2dLayer(3,64,"Padding","same","DilationFactor",1,"Name","rb1_conv1")
    batchNormalizationLayer("Name","rb1_bn1")
    reluLayer("Name","rb1_relu1")

    convolution2dLayer(3,64,"Padding","same","DilationFactor",1,"Name","rb1_conv2")
    batchNormalizationLayer("Name","rb1_bn2")

    additionLayer(2,"Name","rb1_add")
    reluLayer("Name","rb1_out")

    % --- Residual Block 2 (dilation 2) ---
    convolution2dLayer(3,64,"Padding","same","DilationFactor",2,"Name","rb2_conv1")
    batchNormalizationLayer("Name","rb2_bn1")
    reluLayer("Name","rb2_relu1")

    convolution2dLayer(3,64,"Padding","same","DilationFactor",2,"Name","rb2_conv2")
    batchNormalizationLayer("Name","rb2_bn2")

    additionLayer(2,"Name","rb2_add")
    reluLayer("Name","rb2_out")

    % --- Residual Block 3 (dilation 4) ---
    convolution2dLayer(3,64,"Padding","same","DilationFactor",4,"Name","rb3_conv1")
    batchNormalizationLayer("Name","rb3_bn1")
    reluLayer("Name","rb3_relu1")

    convolution2dLayer(3,64,"Padding","same","DilationFactor",4,"Name","rb3_conv2")
    batchNormalizationLayer("Name","rb3_bn2")

    additionLayer(2,"Name","rb3_add")
    reluLayer("Name","rb3_out")

    % --- Residual Block 4 (dilation 8) ---
    convolution2dLayer(3,64,"Padding","same","DilationFactor",8,"Name","rb4_conv1")
    batchNormalizationLayer("Name","rb4_bn1")
    reluLayer("Name","rb4_relu1")

    convolution2dLayer(3,64,"Padding","same","DilationFactor",8,"Name","rb4_conv2")
    batchNormalizationLayer("Name","rb4_bn2")

    additionLayer(2,"Name","rb4_add")
    reluLayer("Name","rb4_out")

    % --- Output ---
    convolution2dLayer(1,1,"Padding","same","Name","conv_out")
    regressionLayer("Name","reg")
];


%% SKIP CONNECTIONS
lgraph = layerGraph(layers);

lgraph = connectLayers(lgraph,"relu_in","rb1_add/in2");
lgraph = connectLayers(lgraph,"rb1_out","rb2_add/in2");
lgraph = connectLayers(lgraph,"rb2_out","rb3_add/in2");
lgraph = connectLayers(lgraph,"rb3_out","rb4_add/in2");


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
fprintf("\Evaluating training...\n");
metrics = evaluate_UNET_scielab(net, dsVal, meanY, stdY);
