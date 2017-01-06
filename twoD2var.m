function abs_freq_result_var = twoD2var( abs_freq_result )

%{
mean = sum( ampitude*freq_ind ) / sum( ampitude );
var = sum( pow( ampitude-mean, 2 ) .* ampitude )/ sum( ampitude );
%}
%freq_ind = [ -15:-1 1:15 ];
freq_ind = [ 16:30 1:15 ];
abs_freq_result_var = zeros( 1148, 8 );
for ant_i=1:8
    %[ tmp, tmp_idx ] = sort( abs_freq_result(: , :, ant_i) );
    for i=1:1148
        tmp  = abs_freq_result( :, i, ant_i );
        mean  = sum( freq_ind*tmp ) / sum( tmp );
        var = sum( (tmp-mean).*(tmp-mean) .*tmp )/sum(tmp);
        result(i) = var;
        %result = maFilter( result );
        %plot(result);d
    end
    abs_freq_result_var( :, ant_i ) = result;
end

end