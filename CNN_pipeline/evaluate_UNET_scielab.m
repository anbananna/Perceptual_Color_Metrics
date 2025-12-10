function metrics = evaluate_UNET_scielab(net, dsVal, meanY, stdY)
% Evaluate trained U-Net model by calculating R, MSE, and RMSE

    Y_true = [];
    Y_pred = [];
    
    reset(dsVal);
    while hasdata(dsVal)
        sample = read(dsVal);   % {X, Ynorm}
        X = sample{1};
        Ynorm = sample{2};
    
        Y_hat_norm = predict(net, X);
    
        % Denormalize BOTH to match original SCIELAB scale
        Y = Ynorm * stdY + meanY;
        Y_hat = Y_hat_norm * stdY + meanY;
    
        Y_true = [Y_true; Y(:)];
        Y_pred = [Y_pred; Y_hat(:)];
    end
    
    metrics.R = corr(Y_true, Y_pred);
    metrics.MSE = mean((Y_pred - Y_true).^2);
    metrics.RMSE = sqrt(metrics.MSE);
    
    fprintf("\n=== Evaluation Metrics ===\n");
    fprintf("R     = %.4f\n", metrics.R);
    fprintf("MSE   = %.6f\n", metrics.MSE);
    fprintf("RMSE  = %.6f\n", metrics.RMSE);

end
