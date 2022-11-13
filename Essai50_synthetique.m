% Ce programme lit plusieurs réponses impulsionnelles de la structure (d'un même fichier) et calcul les moyennes des DSP et
% des moyennes temporelles. Une excitation est alors crèée comme la périodisation d'un triangle pondéré
% par une fenêtre de Hanning : 'signal source'. Enfin la réponse impulsionnelle de la structure est
% convoluée avec le signal source pour créer une réponse synthétique (en fait plusieurs réponses qui seront moyennées).
%;
clear('all');
Fs = 50000*2;              %   fréquence d'échantillonnage en Hz
Duree = 10;		   %Durée d'acquisition en seconde    
L = Duree*Fs;                %   longueur du signal (en échantillons de temps)
t = (1:L) / Fs;                %  échelle temporelle
%;
M = load ('Signal synthetique\09122011+14_03/Signal_0.sig');
figure(1);
%plot(t,s);
plot(t,M(:,2));
xlabel('Time (seconds)'); ylabel('Time waveform');
%
M1 = M(:,2);
M2 = M1(166274:167297);
t1 = (1:1024)/Fs;
figure(101);
plot(t1,M2);
xlabel('Time (seconds)'); ylabel('Time waveform');
%;
S = 2*abs(fft( M2,1024));
S2 = S.^2;
w = (0:511)/512*(Fs/2);
figure(2);
semilogy(w,[S2(1:512)']);
xlabel('Frequency (Hz)'); ylabel('Mag. of Fourier transform');
grid;
%;
%;
M22 = M1(453002:454025);
t1 = (1:1024)/Fs;
figure(102);
plot(t1,M22);
xlabel('Time (seconds)'); ylabel('Time waveform');
%
S12 = 2*abs(fft( M22,1024));
S22 =(S2 +  S12.^2)./2;
w = (0:511)/512*(Fs/2);
figure(202);
semilogy(w,[S22(1:512)']);
xlabel('Frequency (Hz)'); ylabel('Mag. of Fourier transform');
grid;
%;
M23 = M1(734441:735464);
t1 = (1:1024)/Fs;
figure(103);
plot(t1,M23);
xlabel('Time (seconds)'); ylabel('Time waveform');
%;
S13 = 2*abs(fft( M23,1024));
S23 =(S2 + S22 +  S13.^2)./3;
w = (0:511)/512*(Fs/2);
figure(203);
semilogy(w,[S23(1:512)']);
xlabel('Frequency (Hz)'); ylabel('Mag. of Fourier transform');
grid;
%;
%;
Mmoyen =( M2 + M22 + M23) ./3;
figure(104)
plot(t1,Mmoyen);
xlabel('Time (seconds)'); ylabel('Time waveform');
%;
%;
Smoyen = 2*abs(fft( Mmoyen,1024));
Smoyen2 = Smoyen.^2;
w = (0:511)/512*(Fs/2);
figure(204);
semilogy(w,[Smoyen2(1:512)']);
xlabel('Frequency (Hz)'); ylabel('Mag. of Fourier transform');
grid;
%
P = L/1250;
Ex = zeros(1 , L);	%Création du signal d'excitation : triangle pondéré par Hanning
%;
%;
Han = hanning(19);
%
for j = 1 : 10;
for i = 1 : P;
Ex (1 + j + ( i-1 )*1250) = (j/10)* Han(j);
end;
end;
%;
for j = 11 : 19;
for i = 1 : P;
Ex (1 + j + ( i-1 )*1250) =((20 - j)/10)* Han(j);
end;
end;
%;
qbruit =0.2;
%
Exb  = Ex + qbruit* randn(1 , L);        % Signal d'excitation +  bruit gaussien : 5 signaux d'excitation bruités sont créés.
Exb1 = Ex +qbruit* randn(1 , L);	      %Rajouté
Exb2 = Ex +qbruit* randn(1 , L);	      %Rajouté
Exb3 = Ex +qbruit* randn(1 , L);	      %Rajouté
Exb4 = Ex +qbruit* randn(1 , L);	      %Rajouté
%;
figure(3);
plot(t,Exb);
xlabel('Time (seconds)'); ylabel('Time waveform');
%;
NFFT = 2^nextpow2 (  L );                 % Next power of 2 from length of L
%;
SE = 2*abs(fft(Exb, NFFT));
SE2 = SE.^2;
w1 = (0 : NFFT/2 + 1) / (NFFT/2 + 1)*(Fs/2);
figure(4);
semilogy(w1,[SE2(1:NFFT/2 + 2)']);
xlabel('Frequency (Hz)'); ylabel('Mag. of Fourier transform');
grid;
%;
Syn = conv(Exb ,Mmoyen);
Syn1=conv(Exb1,Mmoyen);		 %Rajouté
Syn2=conv(Exb2,Mmoyen);		 %Rajouté
Syn3=conv(Exb3,Mmoyen);		 %Rajouté
Syn4=conv(Exb4,Mmoyen);		 %Rajouté
%NFFT1 = 2^nextpow2 (  L + 1024 );                 % Next power of 2 from length of ( L + 1024 ) ;  pas utile pour 1024 échantillons en plus...
%;
SSyn =2*abs( fft(Syn,NFFT) );
SSyn2 = SSyn.^2;
figure(5);
semilogy(w1,[SSyn2(1:NFFT/2 + 2)']);
grid;
%--------------------------------------------------------------------------------------------------
% A partir d'ici on traite le signal synthétique comme on le ferait pour un signal réel.;
%;
Fs = 50000*2;
t = (1:L)/Fs;
M(:,2) = Syn(1 : L);
M31(:,2)= Syn1(1:L);		 %Rajouté
M32(:,2)= Syn2(1:L);		 %Rajouté
M33(:,2)= Syn3(1:L);		 %Rajouté
M34(:,2)= Syn4(1:L);		 %Rajouté
%
figure(6);
plot(t,M(:,2));
xlabel('Time (seconds)'); ylabel('Time waveform');
S = 2*abs(fft(M(:,2),NFFT));
S2 = S.^2;
w = (0:NFFT/2 +1)/ (NFFT/2 + 1)*(Fs/2);
figure(7);
semilogy(w,[S2(1:NFFT/2 + 2)']);
xlabel('Frequency (Hz)'); ylabel('Mag. of Fourier transform');
grid;
[b,a] = ellip(4,0.1,40,[35000 37000]*2/Fs);
[H,w] = freqz(b,a,512);
figure(8);
plot(w*Fs/(2*pi),abs(H));
xlabel('Frequency (Hz)'); ylabel('Mag. of frequency response');
grid;
MM(:,2) =( M(:,2) + M31(:,2) + M32(:,2) + M33(:,2) + M34(:,2))./5;		%Rajouté
sf = filter(b,a,MM(:,2));					%Rajouté
figure(9);
plot(t,sf);
xlabel('Time (seconds)');
ylabel('Time waveform');
%axis([0 1 -1 1]);
SF = 2*abs(fft(sf,NFFT));
SF2 = SF.^2;
w = (0:NFFT/2 +1)/ (NFFT/2 + 1)*(Fs/2);
figure(10);
semilogy(w,[SF2(1:NFFT/2 + 2)']);
xlabel('Frequency (Hz)'); ylabel('Mag. of Fourier transform');
grid;
H=abs(( hilbert(sf)));
figure(11);
plot(t,H);
E = 2*abs(fft(H,NFFT));
%figure(12);
%semilogy(w,[E(1:NFFT/2 + 2)']);
%			On supprime le 1er échantillon de l'échelle des fréquences (composante continue)
wprime = (1:NFFT/2 +1)/ (NFFT/2 + 1)*(Fs/2);
figure(13);
plot(wprime,[E(2:NFFT/2 + 2)']);
%