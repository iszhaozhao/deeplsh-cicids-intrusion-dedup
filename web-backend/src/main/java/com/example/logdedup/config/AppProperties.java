package com.example.logdedup.config;

import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "app")
public class AppProperties {

    private final Cors cors = new Cors();
    private final Python python = new Python();

    public Cors getCors() {
        return cors;
    }

    public Python getPython() {
        return python;
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
        private String workdir;
        private String script;
        private String modelArtifact;
        private String flowsCsv;

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

        public String getFlowsCsv() {
            return flowsCsv;
        }

        public void setFlowsCsv(String flowsCsv) {
            this.flowsCsv = flowsCsv;
        }
    }
}
