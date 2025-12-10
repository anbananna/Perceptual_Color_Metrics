function out = xyzToPatchPairs(refXYZ, distXYZ, patchDeg, overlap)

    %fprintf("\n--- xyzToPatchPairs DEBUG ---\n");
    %fprintf("PatchDeg = %.2f  overlap = %.2f\n", patchDeg, overlap);
    %fprintf("refXYZ size: %dx%dx3\n", size(refXYZ.xyz,1), size(refXYZ.xyz,2));
    %fprintf("distXYZ size: %dx%dx3\n", size(distXYZ.xyz,1), size(distXYZ.xyz,2));

    %% --- Extract raw patches ---
    refPatches  = splitSceneIntoPatches(refXYZ.xyz,  patchDeg, refXYZ.hfov, overlap);
    distPatches = splitSceneIntoPatches(distXYZ.xyz, patchDeg, distXYZ.hfov, overlap);

    %fprintf("refPatches count: %d\n", size(refPatches,4));
    %fprintf("distPatches count: %d\n", size(distPatches,4));

    numPatches = size(refPatches, 4);
    out = cell(numPatches,1);

    for ii = 1:numPatches

        Rp = refPatches(:,:,:,ii);   % reference patch XYZ
        Dp = distPatches(:,:,:,ii);  % distorted patch XYZ

        Rp = Rp(1:64, 1:64, :);
        Dp = Dp(1:64, 1:64, :);

        % Combine into 6-channel input
        X = cat(3, Rp, Dp);   % (64,64,6)

        % Compute S-CIELAB ΔE map
        dE = computeSCIELABPatch(Rp, Dp, 'D65');
        dE = imresize(dE, [64 64]);

        % Build final output sample
        Y = reshape(dE, [64 64 1]);

        % Store
        out{ii} = { single(X), single(Y) };

    end

    out = out(~cellfun('isempty',out));
end
