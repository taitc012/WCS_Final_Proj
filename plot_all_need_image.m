function plot_all_need_image(final_result, overlap_result, normalized_final_result, normalizedOverlap, dirName,  cf)
[no_use, N_forward_times] = size( final_result );
tic;
cf = cf+1;
figure(cf);
plot( [ [-50:-1] [1:50] ], final_result( :, 1 ) );
title( '100-subcarrier' );
savefig( strcat( dirName, '100-subcarrier' ) );

cf = cf+1;
figure(cf);
[no_use, max_idx] = max( final_result );
plot( max_idx-50, '-ro' );
title( 'max-frequency-time' );
savefig( strcat( dirName, 'max-frequency-time' ) );

cf = cf+1;
figure(cf);
image( [ 0:N_forward_times ], [-50:50], final_result );
title( 'time-frequency' );
savefig( strcat( dirName, 'time-frequency' ) );

cf = cf+1;
figure(cf);
image( [ 0:N_forward_times ], [-50:50], normalized_final_result );
title( 'time-frequency' );
savefig( strcat( dirName, 'time-frequency(normalized)' ) );

cf = cf+1;
figure(cf);
plot( [ [-50:-1] [1:50] ], overlap_result( :, 1 ) );
title( 'overlap-100-subcarrier' );
savefig( strcat( dirName, 'overlap-100-subcarrier' ) );

cf = cf+1;
figure(cf);
image( [ 0:N_forward_times ], [-50:50], normalizedOverlap );
title( 'overlap-time-frequency' );
savefig( strcat( dirName, 'overlap-time-frequency(normalized)' ) );
toc;