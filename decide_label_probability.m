%use confidence(probablility) determine the label of two model

for i=1:5
% 1_test_ra_1_2_3_tx_1_2.scale
[labels instances]=libsvmread(['/home/mvnl/gesture/program/toolbox/libsvm-3.20/' int2str(i) '_test_ra_1_2_3_tx_1_2.scale']);
%labels
% tx_1 model predict result

filename = ['/home/mvnl/gesture/program/toolbox/libsvm-3.20/' int2str(i) '_test_ra_1_2_3_tx_1_2_model_tx_1.out'];
fid = fopen( filename );
test_labels_tx_1 = zeros( length(labels), 2 );
tmp = fgets( fid ); % remove the first line for label 5 3 2 1
run_idx = 1;
while( run_idx<length(labels) )
    tmp = fgets( fid );
    linedata = sscanf( tmp, '%d %lf %lf %lf %lf' );
    test_labels_tx_1( run_idx, 1 ) = linedata(1);
    test_labels_tx_1( run_idx, 2 ) = max( linedata(2:end) );
    run_idx = run_idx+1;
end

% tx_2 model predict result
filename = ['/home/mvnl/gesture/program/toolbox/libsvm-3.20/' int2str(i) '_test_ra_1_2_3_tx_1_2_model_tx_2.out'];
fid = fopen( filename );
test_labels_tx_2 = zeros( length(labels), 2 );
tmp = fgets( fid ); % remove the first line for label 5 3 2 1
run_idx = 1;
while( run_idx<length(labels) )
    tmp = fgets( fid );
    linedata = sscanf( tmp, '%d %lf %lf %lf %lf' );
    test_labels_tx_2( run_idx, 1 ) = linedata(1);
    test_labels_tx_2( run_idx, 2 ) = max( linedata(2:end) );
    run_idx = run_idx+1;
end

final_labels = zeros( size(labels) );
for i=1:length(labels)
    if( test_labels_tx_1( i, 2 ) > test_labels_tx_2( i, 2 ) )
        final_labels(i) = test_labels_tx_1( i, 1 );
    else
        final_labels(i) = test_labels_tx_2( i, 1 );
    end
end

[i sum( final_labels==labels )/length(labels)]

end