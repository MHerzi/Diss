function results = run(data, config)
%RUN Compatibility entry point for the dissertation experiment engine.

arguments
    data
    config (1, 1) struct = diss.config.defaults()
end

results = diss.experiment.run(data, config);

end
