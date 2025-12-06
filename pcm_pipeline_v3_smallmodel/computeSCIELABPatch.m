function dE = computeSCIELABPatch(refXYZ, distXYZ, whitePointName)
% Compute S-CIELAB ΔE map between reference and distorted XYZ patches

    if nargin < 3
        whitePointName = 'D65';
    end

    % Convert whitepoint
    wp = whitepoint(lower(whitePointName));

    % Convert XYZ → linear sRGB
    refRGB  = xyz2srgb(refXYZ);
    distRGB = xyz2srgb(distXYZ);

    % Clamp to valid range
    refRGB  = max(min(refRGB,1),0);
    distRGB = max(min(distRGB,1),0);

    % Compute S-CIELAB ΔE map
    dE = scielab(refRGB, distRGB, wp);

    % Ensure numeric stability
    dE(isnan(dE)) = 0;
end
