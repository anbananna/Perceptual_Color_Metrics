function visualize_example(X, Y_gt, Y_hat, id)
    % VISUALIZE_EXAMPLE
    % Visualize XYZ reference/distorted patches and S-CIELAB ΔE maps.
    
    figure('Name', sprintf('Example %d', id), 'NumberTitle', 'off');
    
    % Extract XYZ channels
    refXYZ  = X(:,:,1:3);
    distXYZ = X(:,:,4:6);
    
    % Ensure XYZ values are within valid range [0,1] to avoid warnings
    refXYZ_clipped  = max(min(refXYZ,1),0);
    distXYZ_clipped = max(min(distXYZ,1),0);
    
    % Convert to RGB for visualization
    try
        refRGB  = xyz2rgb(refXYZ_clipped);
        distRGB = xyz2rgb(distXYZ_clipped);
    catch
        % If xyz2rgb is unavailable (older MATLAB), show XYZ directly
        refRGB  = repmat(refXYZ_clipped(:,:,2),1,1,3);   % use Y channel as grayscale
        distRGB = repmat(distXYZ_clipped(:,:,2),1,1,3);
    end
    
    % --- Plot Reference RGB ---
    subplot(2,3,1);
    imshow(refRGB);
    title('Reference Patch');
    
    % --- Plot Distorted RGB ---
    subplot(2,3,2);
    imshow(distRGB);
    title('Distorted Patch');
    
    % --- Plot RGB difference ---
    subplot(2,3,3);
    imshow(abs(refRGB - distRGB), []);
    title('RGB Difference');
    
    % --- Ground truth ΔE ---
    subplot(2,3,4);
    imagesc(Y_gt);
    axis image off;
    colorbar;
    title('Ground Truth \DeltaE');
    
    % --- Predicted ΔE ---
    subplot(2,3,5);
    imagesc(Y_hat);
    axis image off;
    colorbar;
    title('Predicted \DeltaE');
    
    % --- Absolute error ---
    subplot(2,3,6);
    imagesc(abs(Y_gt - Y_hat));
    axis image off;
    colorbar;
    title('Absolute Error');
    
    colormap jet;

end
