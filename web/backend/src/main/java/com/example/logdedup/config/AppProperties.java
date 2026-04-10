package com.example.logdedup.config;

import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "app")
public class AppProperties {

    private final Cors cors = new Cors();
    private final Python python = new Python();
    private final Experiments experiments = new Experiments();

    public Cors getCors() {
        return cors;
    }

    public Python getPython() {
        return python;
    }

    public Experiments getExperiments() {
        return experiments;
    }

    public static class Cors {
        private String allowedOrigins = "http://localhost:5173";

        public String getAllowedOrigins() {
            return allowedOrigins;
        }

        public void setAllowedOrigins(String allowedOrigins) {
            this.allowedOrigins = allowedOrigins;
        }
    }

    public static class Python {
        private String command = "python3";
        private String workdir;
        private String script;
        private String modelArtifact;
        private String mlpModelArtifact;
        private String bigruModelArtifact;
        private String flowsCsv;

        public String getCommand() {
            return command;
        }

        public void setCommand(String command) {
            this.command = command;
        }

        public String getWorkdir() {
            return workdir;
        }

        public void setWorkdir(String workdir) {
            this.workdir = workdir;
        }

        public String getScript() {
            return script;
        }

        public void setScript(String script) {
            this.script = script;
        }

        public String getModelArtifact() {
            return modelArtifact;
        }

        public void setModelArtifact(String modelArtifact) {
            this.modelArtifact = modelArtifact;
        }

        public String getMlpModelArtifact() {
            return mlpModelArtifact;
        }

        public void setMlpModelArtifact(String mlpModelArtifact) {
            this.mlpModelArtifact = mlpModelArtifact;
        }

        public String getBigruModelArtifact() {
            return bigruModelArtifact;
        }

        public void setBigruModelArtifact(String bigruModelArtifact) {
            this.bigruModelArtifact = bigruModelArtifact;
        }

        public String getFlowsCsv() {
            return flowsCsv;
        }

        public void setFlowsCsv(String flowsCsv) {
            this.flowsCsv = flowsCsv;
        }
    }

    public static class Experiments {
        private String resultsDir;

        public String getResultsDir() {
            return resultsDir;
        }

        public void setResultsDir(String resultsDir) {
            this.resultsDir = resultsDir;
        }
    }
}
