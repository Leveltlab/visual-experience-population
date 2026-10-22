args <- commandArgs(trailingOnly=TRUE)
root <- if(length(args)) args[1] else "results/spikes/manuscript_statistics"
.libPaths(c(file.path(root,"R-library"),.libPaths()))
library(nparLD)
stopifnot(as.character(packageVersion("nparLD"))=="2.3.3")
options(digits=16)
results <- list()
for(panel in c("B","C","F","G")) {
 d <- read.csv(file.path(root,paste0("input_",panel,".csv")))
 d$mouse <- factor(d$mouse)
 d$rearing <- factor(d$rearing,levels=c("DR","NR"))
 d$area <- factor(d$area,levels=c("V1","AM-PM","A-AL-RL","LM"))
 d$time <- factor(d$time,levels=c("baseline","chronic"))
 d$level <- factor(d$level,levels=switch(panel,B=c("90","180"),C=c("direction","orientation"),F=c("1","2"),G=c("1","2","3")))
 # A full crossed four-factor marginal-effects design. Retain only the four requested main effects.
 fit <- nparLD(response ~ level * area * rearing * time, data=d, subject="mouse", hypothesis="H0p", effect="unweighted", cell.weights="subjects", covariance=TRUE, Factor.Information=TRUE)
 saveRDS(fit,file.path(root,paste0("fit_",panel,".rds")))
 write.csv(fit$.internal$Sigma,file.path(root,paste0("covariance_",panel,".csv")),row.names=FALSE)
 tab <- as.data.frame(ATS(fit));tab$factor<-rownames(tab);tab$panel<-panel;rownames(tab)<-NULL
 tab <- tab[tab$factor %in% c("level","area","rearing","time"),]
 # Numerically stable upper tail; package's 1-pf rounds tiny probabilities to zero.
 tab$log_p <- pf(tab$Statistic,tab$df,1e10,lower.tail=FALSE,log.p=TRUE)
 tab$p_raw <- exp(tab$log_p)
 tab$p_package <- tab[["p-value"]];tab[["p-value"]]<-NULL
 results[[panel]]<-tab
 capture.output(list(Design=fit$Design,wholeplots=fit$wholeplots,subplots=fit$subplots,N=fit$N.info,main_effects=tab),file=file.path(root,paste0("model_",panel,".txt")))
 for(term in c("level","area","rearing","time"))write.csv(fit$factor.info[[term]],file.path(root,paste0("marginal_",panel,"_",term,".csv")))
 write.csv(effects(fit),file.path(root,paste0("relative_effects_",panel,".csv")),row.names=FALSE)
 cat("\nPANEL",panel,"\n");print(tab);cat("wholeplots:",fit$wholeplots,"subplots:",fit$subplots,"\n")
}
r <- do.call(rbind,results);rownames(r)<-NULL
write.csv(r,file.path(root,"factorial_ats_raw.csv"),row.names=FALSE)
capture.output(sessionInfo(),file=file.path(root,"R_session_info.txt"))
