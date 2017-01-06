function [ step, f ] = music(rx_ant, n_signal, n_ant, lambda, d_interspacing, n_samples, mean_noise )
%[num_ant num_samples]=size(rx_ant)

SHOW_FIG = 0;


%rx_ant received signal
%disp('Start MUSIC algorithm');

half_wl = lambda/2;
kp=2*pi/lambda;
d= d_interspacing;
x = n_ant;
num_ant = n_ant;
%tic;
U=rx_ant;

%%spatial smoothing
Ng = 1;
U = spatial_smoothing( U, Ng );
[num_ant, n_samples] = size( U );

conoise_mat=zeros(num_ant,num_ant);
%size(conoise_mat)
%{
for i=1:n_samples;
    covar=U(:,i)*(U(:,i)');
    %size(covar)
    conoise_mat=conoise_mat+covar;
end
conoise_mat=conoise_mat/n_samples;
%}

conoise_mat = U*U'/n_samples;
%{
tic;
g_U = gpuArray( U );
g_Ut = gpuArray( U' );
g_conoise_mat = mtimes( g_U, g_Ut );
conoise_mat = gather( g_conoise_mat );
conoise_mat = conoise_mat/n_samples;
toc;
%}
%{
if( conoise_mat == conoise_mat_ori )
    'yes'
else
    'no'
    abs(conoise_mat)-abs(conoise_mat_ori)
end
return
%}

[eigen_vec,eigen_val]=eig(conoise_mat,'nobalance');
%-------------------------------------------
% The estimated incoming signal 
%-------------------------------------------
[ sort_value, sort_idx ] = sort( diag(eigen_val) );
%sort_value
mean_noise = 0.01*sort_value(end);
[find_idx] = find( sort_value< 0.1*sort_value(end) );
%[find_idx] = find( sort_value<0.1*sort_value(end) | sort_value<0.05);
%[find_idx] = find( sort_value<0.0003 );
noise_mat = eigen_vec( :, sort_idx( find_idx ) );
%size(noise_mat)


%{
for i=1:num_ant
    value(i)=eigen_val(i,i);
end
sort_value=sort(value);
increase=0;
mean_noise = 1/( 10^(SNR/10) );

for i=1: num_ant
    if sort_value(i)<1.5*mean_noise
        increase=increase+1;
    end
end
noise_mat=zeros(num_ant,increase);
signal_mat=zeros(num_ant,num_ant);
for i=1:num_ant
    for k=1:num_ant
        if sort_value(i)== value(k)
            signal_mat(:,i)=eigen_vec(:,k);
        end
    end
end
for i=1: increase
    noise_mat(:,increase-i+1)=signal_mat(:,i);
end
%}

%disp(' ')
%gst=[' Number of antennas=' int2str(num_ant)];
%disp(gst)
if( SHOW_FIG == 1 )
    cf = cf+1;
    figure(cf);
    plot([1:num_ant],abs((fliplr(sort_value))),'-+');
    grid on
    hold on
    plot([1:num_ant],mean_noise*[ones(1,num_ant)],'-o');
    legend('extract value','noise')
    xlabel('Antennas');
    ylabel('Values');
    title('The magnitude of the received signal matrix relationship');
end

q1=0;
q2=180;
q3=q2-q1;
f=zeros(1,101);
i=1;
for step= q1:q3/100:q2
    k=0:num_ant-1;
    dist=exp(j.*k*kp*d*cos(step*pi/180));
    a=dist.';
    %f(i)=(a'*a)/(a'*noise_mat*noise_mat'*a);
    f(i)=1/(a'*noise_mat*noise_mat'*a);
    %{
    if( step<15 || step>165 )
        f(i) = f(i)*sin( step*pi/180 );
    end
    %}
    i=i+1;
end
step= q1:q3/100:q2;
f=abs(f)./max(abs(f));
%{
[f_pks, f_idxs] = findpeaks( f );
f_pks;
step(f_idxs);
[pks,idxs] = findpeaks(20*log10(f));
step(idxs);
%}

cf = 2;
    %cf = cf+1;
    figure(cf);
    radians = deg2rad(step);
    polar(radians,f);
    axis tight;
    title( 'AoA probability result' );
if( SHOW_FIG == 1 )
    savefig( [dirName '/' 'AoA-result'] );
end

if(SHOW_FIG == 1)

    %cf = cf+1;
    figure(cf);
    radians = deg2rad(step);
    polar(radians,20*log10(f));
    %axis tight;

    %cf = cf+1;
    figure(cf);
    plot(step,f);
    xlim([-180 180]);
    %ylim([0 1]);
    %axis tight;

    cf = cf+1;
    figure(cf);
    plot(step,20*log10(f));
    grid on
    xlabel('A');
    ylabel('Size(dB)');
    title('MUSIC algorithm the angle estimation');
    axis tight;

end

TOP_NUM = 10;
[max_pks, sort_idx] = sort( f, 'descend' );
%max_pks(1:5)
aa = step(sort_idx(1:TOP_NUM));
%[aa.' max_pks(1:TOP_NUM).']
%{
for aa_i=1:TOP_NUM
    fprintf('%f', aa(aa_i));
end
fprintf( '\n' );
%}
%toc;
