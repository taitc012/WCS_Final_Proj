function ss_result = spatial_smoothing( X, Ng )

[ num_ant, num_samples ] = size(X);
if( Ng>num_ant )
    fprintf('Number of sub-array is larger than number of antennas ');
    ss_result = X;
    return;
end
if( Ng==1 )
    ss_result = X;
    return;
end
ss_result = zeros( num_ant-Ng+1, num_samples );
tmp = zeros( Ng, num_samples );
for i = 1:num_ant-Ng+1
    tmp = X( i:i+Ng-1, : );
    ss_result(i,:) = mean( tmp );
end