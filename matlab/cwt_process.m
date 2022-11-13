
function icwt_signal = cwt_tranform(fs,name,fmin,fmax)

    %{
    Transformée en ondelette continue.
    fs: fréquence d'échantillonage
    name : nom du fichier contenant le signal à traiter
    fmin , fmax : fenêtre de zoom suite à une transformée de morlet.
    %}
     
     

    %définition des vars
    
    disp(class(fs))
    T = readtable(name);
    signal = T(:,"signal");
    t = T(:,"seconde");
    signal = table2array(signal);
    t = table2array(t);

    %cwt puis cwt inverse ciblée sur la fenêtre de fréquence
    [cfs,f] = cwt(signal,'morse',fs);
    icwt_signal = icwt(cfs,f,[fmin fmax]);
    
    %plots
    subplot(2,1,1)
    plot(signal)
    grid on
    title("Original Data")
    ylabel("Amplitude")
    axis tight
    
    subplot(2,1,2)
    plot(icwt_signal)
    grid on
    title(['Bandpass Filtered Reconstruction [',num2str(fmin),' ',num2str(fmax),'] Hz']);
    xlabel("Time (s)")
    ylabel("Amplitude")
    axis tight
end
