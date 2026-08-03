function collatzplot(m)
% Plot length of sequence for Collatz problem
% Prepare figure
clf
set(gca,'XScale','linear')
%
% Determine and plot sequence and sequence length
for N = 1:m
	plot_seq = collatz(N);
	seq_length(N) = length(plot_seq);
	line(N,plot_seq,'Marker','.','MarkerSize',9,'Color','blue')
	drawnow
end