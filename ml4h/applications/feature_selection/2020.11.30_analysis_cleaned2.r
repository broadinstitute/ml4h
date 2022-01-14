library(data.table)
library(survival)
library(survminer)
library(tableone)
library(ggplot2)
library(GGally)
library(patchwork)
library(gtools)
library(plyr)
library(Hmisc)
library(mltools)
library(RColorBrewer)
library(ResourceSelection)
library(pROC)
library(corrplot)

absolute_risk <- function(coxmodel,data){
    cumhaz <- basehaz(coxmodel,centered = TRUE) #Note that this is centered at mean covariates
    basesurv <- exp(-1*cumhaz$hazard[which.min(abs(cumhaz$time - 3652))])
    print(paste0("S_o(10) = ",round(basesurv,3)))
    
    tenyrpredict <- 1 - basesurv^(predict(coxmodel,data,type="risk"))
    return(tenyrpredict)
}

calibration_plot <- function(data,predictioncol,eventcol,manualpvalue=NULL){
    if(any(is.na(cbind(data[[predictioncol]],data[[eventcol]])))){
        warning('at least 1 NA entry in predictioncol or eventcol')
    }
    
    pred_quantiles <- quantcut(data[[predictioncol]],q=10)
    pred_mean <- tapply(data[[predictioncol]],pred_quantiles,mean)
    event_mean <- tapply(data[[eventcol]],pred_quantiles,mean)
    data.lm <- lm(event_mean ~ pred_mean)
#     print(data.lm)
    
    num <- round_any(max(max(pred_mean),max(event_mean))*1.4,0.05)
#     print(max(pred_mean))
#     print(max(event_mean))
#     print(num)
    
    
    
#     print(pred_mean)
#     print(event_mean)
#     print(cbind(pred_mean,event_mean))
    
    #Greenwood-Nam-D'Agostino
#     gnd_result <- GND.calib(pred=data[[predictioncol]],tvar=pmin(data[[timetoeventcol]],censortime),
#                            out=data[[eventcol]],cens.t=data[[censorcol]],groups=pred_quantiles,
#                            adm.cens=censortime)
#     print(paste0('GND: ',gnd_result))
    
    hoslem_result <- hoslem.test(x=data[[predictioncol]],y=as.numeric(data[[eventcol]]),g=10)
#     print(paste0('HL: ',hoslem_result))
    HLpval <- round(hoslem_result$p.value,3)
    HLpvalprint <- paste0("p-value = ",HLpval)
    
    if(!is.null(manualpvalue)) HLpvalprint <- manualpvalue
    
    g1 <- ggplot(data=as.data.frame(cbind(pred_mean,event_mean)),aes(x=pred_mean,y=event_mean)) + 
    geom_point() + 
    geom_abline(intercept=0,slope=1,linetype="dashed") + 
    geom_abline(intercept=data.lm$coefficients[['(Intercept)']],slope=data.lm$coefficients[['pred_mean']],color="red") +
    scale_x_continuous(limits=c(0,num),breaks=seq(0,num,by=0.05)) +
    scale_y_continuous(limits=c(0,num),breaks=seq(0,num,by=0.05)) + 
    xlab("Predicted 10-year Cumulative Incidence") +
    ylab("Observed 10-year Cumulative Incidence") +
    theme_classic() +
    theme(text = element_text(size=16)) +
    annotate("text",x=0.8*num,y=0.2*num,size=6,label=paste0("m = ",round(data.lm$coefficients[['pred_mean']],2))) +
    annotate("text",x=0.8*num,y=0.15*num,size=6,label=paste0("b = ",round(data.lm$coefficients[['(Intercept)']],3))) +
    annotate("text",x=0.8*num,y=0.10*num,size=6,label=HLpvalprint)
    return(g1)
}

reclass <- function(data,cOutcome,predrisk1,predrisk2,cutoff){
 
c1 <- cut(predrisk1,breaks = cutoff ,include.lowest=TRUE,right= FALSE)
c2 <- cut(predrisk2,breaks = cutoff ,include.lowest=TRUE,right= FALSE)
tabReclas <- table("Initial Model"=c1, "Updated Model"=c2)
cat(" _________________________________________\n")
cat(" \n     Reclassification table    \n")
cat(" _________________________________________\n")
 
 ta<- table(c1, c2, data[[cOutcome]])
    print(ta)
 
  cat ("\n Outcome: absent \n  \n" )
  TabAbs <- ta[,,1]
  tab1 <- cbind(TabAbs, " % reclassified"= round((rowSums(TabAbs)-diag(TabAbs))/rowSums(TabAbs),2)*100)
  names(dimnames(tab1)) <- c("Initial Model", "Updated Model")
  print(tab1)
 
  cat ("\n \n Outcome: present \n  \n" )
  TabPre <- ta[,,2]
  tab2 <- cbind(TabPre, " % reclassified"= round((rowSums(TabPre)-diag(TabPre))/rowSums(TabPre),2)*100)
  names(dimnames(tab2)) <- c("Initial Model", "Updated Model")
  print(tab2)
 
  cat ("\n \n Combined Data \n  \n" )
  Tab <- tabReclas
  tab <- cbind(Tab, " % reclassified"= round((rowSums(Tab)-diag(Tab))/rowSums(Tab),2)*100)
  names(dimnames(tab)) <- c("Initial Model", "Updated Model")
  print(tab)
cat(" _________________________________________\n")
 
c11 <-factor(c1, levels = levels(c1), labels = c(1:length(levels(c1))))
c22 <-factor(c2, levels = levels(c2), labels = c(1:length(levels(c2))))
 
  x<-improveProb(x1=as.numeric(c11)*(1/(length(levels(c11)))),
  x2=as.numeric(c22)*(1/(length(levels(c22)))), y=data[[cOutcome]])
 
 
y<-improveProb(x1=predrisk1, x2=predrisk2, y=data[[cOutcome]])

 
cat("\n NRI(Categorical) [95% CI]:", round(x$nri,4),"[",round(x$nri-1.96*x$se.nri,4),"-",
 round(x$nri+1.96*x$se.nri,4), "]", "; p-value:", round(2*pnorm(-abs(x$z.nri)),5), "\n" )
 
 cat(" NRI(Continuous) [95% CI]:", round(y$nri,4),"[",round(y$nri-1.96*y$se.nri,4),"-",
 round(y$nri+1.96*y$se.nri,4), "]", "; p-value:", round(2*pnorm(-abs(y$z.nri)),5), "\n" )
    
cat(" NRIevent(Continuous) [95% CI]:", round(y$nri.ev,4),"[",round(y$nri.ev-1.96*y$se.nri.ev,4),"-",
 round(y$nri.ev+1.96*y$se.nri.ev,4), "]", "; p-value:", round(2*pnorm(-abs(y$z.nri.ev)),5), "\n" )
    
cat(" NRInonevent(Continuous) [95% CI]:", round(y$nri.ne,4),"[",round(y$nri.ne-1.96*y$se.nri.ne,4),"-",
 round(y$nri.ne+1.96*y$se.nri.ne,4), "]", "; p-value:", round(2*pnorm(-abs(y$z.nri.ne)),5), "\n" )
 
cat(" IDI [95% CI]:", round(y$idi,4),"[",round(y$idi-1.96*y$se.idi,4),"-",
 round(y$idi+1.96*y$se.idi,4), "]","; p-value:", round(2*pnorm(-abs(y$z.idi)),5), "\n")
    
#     return(y)
}


#Master data.table with UKB ids and many descriptive variables relevant to PCE
masterdfukb <- fread("/medpop/esp2/sagrawal/prs_crs_integration/prs_crs_masterdf_ascvd_sbpavg.csv")

#Hard CAD
CAD_SA <- fread("CAD_SA.csv")

#CVD outcome
CVD_Elliot <- fread('/medpop/esp2/sagrawal/prs_crs_integration/Elliot_CVD_analysisready.csv')

#QRISK3 for UKBB
qrisk <- fread("/medpop/esp2/aniruddh/SouthAsians/Qriskscoresnonmissingnew.txt")
qrisk$qrisk3 <- qrisk$QRISK3_2017/100

#Processing Hard CAD outcome file to include 10 year outcome
CAD_SA <- CAD_SA[,c("eid","incident_disease","fu_days","fu_yrs")]
names(CAD_SA) <- c("eid","CAD_incident_disease","CAD_fu_days","CAD_fu_yrs")
CAD_SA$CAD_tenyearoutcome <- 0
CAD_SA[CAD_incident_disease == 1 & 
       CAD_fu_days <= 3652.5]$CAD_tenyearoutcome <- 1
CAD_SA$CAD_tenyearcensor <- ifelse(CAD_SA$CAD_fu_days > 3652.5,1,0)

CVD_Elliot <- CVD_Elliot[,c("eid","incident_disease","fu_days","fu_yrs")]
names(CVD_Elliot) <- c("eid","CVD_incident_disease","CVD_fu_days","CVD_fu_yrs")
CVD_Elliot$CVD_tenyearoutcome <- 0
CVD_Elliot[CVD_incident_disease == 1 & 
       CVD_fu_days <= 3652.5]$CVD_tenyearoutcome <- 1
CVD_Elliot$CVD_tenyearcensor <- ifelse(CVD_Elliot$CVD_fu_days > 3652.5,1,0)

masterdf1 <- merge(masterdfukb,CAD_SA,by="eid")
masterdf1 <- merge(masterdf1,CVD_Elliot,by="eid")
masterdf <- merge(masterdf1,qrisk[,c('ID','qrisk3')],by.x='eid',by.y='ID',all=TRUE)

#Groups made of self-reported race/ethnicity
whiteonlyvec <- c("Any_other_white_background","British","Irish","White")
southasianvec <- c("Bangladeshi","Indian","Pakistani")
asianminusSA <- c("Any_other_Asian_background","Asian_or_Asian_British","Chinese","White_and_Asian")
blackvec <- c("African","Any_other_Black_background","Black_or_Black_British","Caribbean","White_and_Black_African","White_and_Black_Caribbean")
othervec <- c("Any_other_mixed_background","Mixed","Other_ethnic_group","Do_not_know","Prefer_not_to_answer") #is.na in this too in terms of PC similarity

masterdf$sex_strat <- ifelse(masterdf$sex == "Male",1,0)

masterdf$ethnicity_white <- ifelse(masterdf$ethnicity %in% whiteonlyvec,1,0)

masterdf$ethnicity_black <- ifelse(masterdf$ethnicity %in% blackvec,1,0)

masterdf$ethnicity_asian <- ifelse(masterdf$ethnicity %in% asianminusSA,1,0)

masterdf$ethnicity_southasian <- ifelse(masterdf$ethnicity %in% southasianvec,1,0)

masterdf$ethnicity_other <- ifelse((masterdf$ethnicity %in% othervec) | is.na(masterdf$ethnicity),1,0)

masterdf$ethnicity_group <- "placeholder"
masterdf[ethnicity_white==1]$ethnicity_group <- "White"
masterdf[ethnicity_black==1]$ethnicity_group <- "Black"
masterdf[ethnicity_asian==1]$ethnicity_group <- "East Asian"
masterdf[ethnicity_southasian==1]$ethnicity_group <- "South Asian"
masterdf[ethnicity_other==1]$ethnicity_group <- "Other"

masterdf$ethnicity_group <- factor(masterdf$ethnicity_group,levels=c('Black','South Asian','East Asian','White','Other'))

# Defined this for the purpose of recalibration

masterdf$age_group <- "G1" #(age < 45, basically meaning 40-45)
masterdf[age >= 45 & age < 50]$age_group <- "G2"
masterdf[age >= 50 & age < 55]$age_group <- "G3"
masterdf[age >= 55 & age < 60]$age_group <- "G4"
masterdf[age >= 60 & age < 65]$age_group <- "G5"
masterdf[age >= 65 & age < 70]$age_group <- "G6"
masterdf[age >= 70 & age < 75]$age_group <- "G7"

masterdf$ageG2 <- ifelse(masterdf$age_group=='G2',1,0)
masterdf$ageG3 <- ifelse(masterdf$age_group=='G3',1,0)
masterdf$ageG4 <- ifelse(masterdf$age_group=='G4',1,0)
masterdf$ageG5 <- ifelse(masterdf$age_group=='G5',1,0)
masterdf$ageG6 <- ifelse(masterdf$age_group=='G6',1,0)
masterdf$ageG7 <- ifelse(masterdf$age_group=='G7',1,0)

#Define a secondary follow-up time variable that cuts off at 10 years
masterdf[CAD_fu_days >= 3652,CAD_fu_days10yr:=3652]
masterdf[CAD_fu_days < 3652,CAD_fu_days10yr:=CAD_fu_days]

#Adjusting PRS for ancestry

pcmod1 <- lm(prs ~ PC1+PC2+PC3+PC4, data=masterdf)
masterdf$prs_resid <- masterdf$prs - predict(pcmod1,masterdf)

##Importing additional descriptive variables I want

ldl <- fread("/medpop/esp2/sagrawal/prs_crs_integration/labs_ldl_ukbb.csv")
names(ldl)[5] <- 'LDL_C'
masterdf <- merge(masterdf,ldl[,c("sample_id","LDL_C")],by.x='eid',by.y='sample_id',all=TRUE)

#Contain the patient ids that went into the development and holdout sets
#Only contains 51 columns, corresponding to 51 nonzero features in Coxnet model
#at selected alpha

fullfeat_dev <- fread("coxnet__feature_matrix__development_0_0011765691116882482__30Nov_2020.txt")
fullfeat_ho <- fread("coxnet__feature_matrix__holdout_0_0011765691116882482__30Nov_2020.txt")

fullfeat_devUnscaled <- fread('coxnet__feature_matrix_nonzero_model_0_0011765691116882482__development_unscaled__30Nov_2020.txt')
fullfeat_hoUnscaled <- fread('coxnet__feature_matrix_nonzero_model_0_0011765691116882482__holdout_unscaled__30Nov_2020.txt')

masterdf$subgroup_indicator <- 0
masterdf[eid %in% fullfeat_dev$index]$subgroup_indicator <- 1
masterdf[eid %in% fullfeat_ho$index]$subgroup_indicator <- 2

masterdf_dev <- masterdf[subgroup_indicator == 1]

masterdf_tab1 <- masterdf

masterdf_tab1$framrisk <- masterdf_tab1$framrisk * 100
masterdf_tab1$tenyearASCVD <- masterdf_tab1$tenyearASCVD * 100
masterdf_tab1$qrisk3 <- masterdf_tab1$qrisk3 * 100
masterdf_tab1$Cholesterol <- masterdf_tab1$Cholesterol * 38.6
masterdf_tab1$HDL_C <- masterdf_tab1$HDL_C * 38.6
masterdf_tab1$LDL_C <- masterdf_tab1$LDL_C * 38.6
masterdf_tab1$prs_resid_scaled <- scale(masterdf_tab1$prs_resid)

masterdf_tab1$included <- ifelse(masterdf_tab1$subgroup_indicator==0,0,1)

tab1vars <- c('age','sex','ethnicity_group','currentsmoker','DM','Cholesterol',
              'HDL_C','LDL_C','systolicbp','antihtnrx','prs_resid_scaled','framrisk',
              'tenyearASCVD','qrisk3','CAD_incident_disease','CAD_tenyearoutcome')
tab1facvars <- c('sex','ethnicity_group','currentsmoker','DM','antihtnrx',
             'CAD_incident_disease','CAD_tenyearoutcome')

tabone=CreateTableOne(tab1vars,data=masterdf_tab1,strata=c('subgroup_indicator'),factorVars=tab1facvars)
print(tabone,quote = F,digits=1)

tabone=CreateTableOne(tab1vars,data=masterdf_tab1[subgroup_indicator != 0],strata=c('subgroup_indicator'),factorVars=tab1facvars)
print(tabone,quote = F,digits=1)

tabone=CreateTableOne(tab1vars,data=masterdf_tab1,strata=c('included'),factorVars=tab1facvars)
print(tabone,quote = F,digits=1)

xgcoxAbsRiskDev <- fread('/medpop/esp2/sagrawal/mi_mlprediction/2020.11.30_marcus/2021.01.12_xgboost2/model2_xgboost_cox_best_model_predictions_all_data__absolute_risk_with_ids__development__model2__30Nov_2020.txt')
xgcoxAbsRiskHo <- fread('/medpop/esp2/sagrawal/mi_mlprediction/2020.11.30_marcus/2021.01.12_xgboost2/model2_xgboost_cox_best_model_predictions_all_data__absolute_risk_with_ids__holdout__model2__30Nov_2020.txt')

xgcoxBaseSurv <- fread('/medpop/esp2/sagrawal/mi_mlprediction/2020.11.30_marcus/2021.01.12_xgboost2/model2_xgboost_cox_best_model_predictions_all_data__baseline_survival__model2__30Nov_2020.txt')
xgcoxLeave1OutLabel <- fread('/medpop/esp2/sagrawal/mi_mlprediction/2020.11.30_marcus/xgcoxFeatLabel.csv')

xgcoxFeatList <- fread('/medpop/esp2/sagrawal/mi_mlprediction/2020.11.30_marcus/2021.01.12_xgboost2/model2_xgboost_cox_best_model_predictions_all_data__feature_importance__total_gain__overall__model2__30Nov_2020.txt')

xgcoxLeave1OutLabel$category <- factor(xgcoxLeave1OutLabel$category,
                                             levels=c("Demographics",
                                                     "Lifestyle",
                                                     "Medical history",
                                                     "Surgical history",
                                                     "Family history",
                                                     "Physical exam",
                                                     "Genetics",
                                                     "Labs"),
                                     ordered=TRUE)

xgcoxLeave1OutLabelOrder <- xgcoxLeave1OutLabel[order(category)]

coxnet_feature_list_annot_clean <- fread('/medpop/esp2/sagrawal/mi_mlprediction/2020.11.30_marcus/coxnet51_annotate.csv')

coxnet_feature_list_annot_clean$category <- factor(coxnet_feature_list_annot_clean$category,
                                             levels=c("Demographics",
                                                     "Lifestyle",
                                                     "Medical history",
                                                     "Surgical history",
                                                     "Family history",
                                                     "Physical exam",
                                                     "Genetics",
                                                     "Labs"),
                                     ordered=TRUE)

simplecox_feat <- fread("/medpop/esp2/sagrawal/mi_mlprediction/2020.11.30_marcus/coxph__feature_list_0_0011765691116882482__30Nov_2020.txt")
simplecox_feat <- simplecox_feat[-1]
names(simplecox_feat) <- c('field','simplecoxcoef')
coxnetvscox <- merge(coxnet_feature_list_annot_clean,simplecox_feat,by='field')
coxnetvscox$coxnetHR <- exp(coxnetvscox$coef)
coxnetvscox$simplecoxHR <- exp(coxnetvscox$simplecoxcoef)
# coxnetvscox[order(category)]

coxnetvscox$simplecoxdiff <- abs(coxnetvscox$coxnetHR-coxnetvscox$simplecoxHR)

coxnet_perm_imp <- fread("/medpop/esp2/sagrawal/mi_mlprediction/2020.11.30_marcus/2021.01.12_misc/coxnet_final_model_permutation_importance__model_0_0002480304066510752.txt")

coxnet_perm_imp <- coxnet_perm_imp[,V1 := NULL]

a <- apply(coxnet_perm_imp,2,mean)
b <- apply(coxnet_perm_imp,2,function(x) quantile(x,c(0.025,0.975)))

coxnet_perm_imp_clean <- as.data.table(rbind(a,b))

coxnet_perm_imp_clean <- transpose(coxnet_perm_imp_clean)

coxnet_perm_imp_clean$field <- names(a)

coxnet_perm_merged <- merge(coxnet_perm_imp_clean,coxnet_feature_list_annot_clean,by='field')
coxnet_perm_merged <- coxnet_perm_merged[order(-V1)]

coxnet_perm_merged$category <- factor(coxnet_perm_merged$category,
                                             levels=c("Demographics",
                                                     "Lifestyle",
                                                     "Medical history",
                                                     "Surgical history",
                                                     "Family history",
                                                     "Physical exam",
                                                     "Genetics",
                                                     "Labs"),
                                     ordered=TRUE)

options(repr.plot.width=12,repr.plot.height=8)

featplot <- ggplot(coxnet_perm_merged[1:20],aes(x=reorder(human,-V1),y=V1,fill=category)) +
geom_bar(stat="identity",color="black") +
geom_errorbar(aes(ymin=V2,ymax=V3),width=0.2) +
xlab("") +
ylab("Predictor importance\n(Leave-one-out C-statistic decrease)") +
labs(fill='Category') +
theme_classic() + 
theme(axis.text.x = element_text(angle=90,hjust=0.95,vjust=0.5,color='black'),text = element_text(size=16,color='black'),
     legend.position=c(0.9,0.5)) +
# scale_fill_discrete(drop=FALSE) +
scale_fill_brewer(palette = 'Set2',drop=FALSE) +
scale_y_continuous(expand=expansion(mult = c(0,0.1))) + 
scale_x_discrete(breaks=coxnet_perm_merged[1:20]$human,
                labels=c('Age',expression(GPS[CAD]),'Sex','HDL-c','SBP','HbA1C',
                         'LDL-c','Testosterone','Hip circ.','ApoB','Waist circ.',
                         'Cystatin C','Father heart dz','Lp(a)','Hypertension',
                         'Fair health rating','Neutrophils','Sibling heart dz',
                         'PC3','Lipid-lowering'))
#the drop=FALSE allows all categories to remain even though I subsetting the datatable

featplot

# ggsave('coxnet51_leave1out.pdf',g2,width=10,height=8)

###Feat plot for xgcox

options(repr.plot.width=12,repr.plot.height=6)

xgCoxFeatPlot <- ggplot(xgcoxLeave1OutLabel[1:20],aes(x=reorder(abbrev,-mean),y=mean,fill=factor(overlap,levels=c(1,0),ordered=TRUE,labels=c('Yes','No')))) +
geom_bar(stat="identity",color="black") +
geom_errorbar(aes(ymin=lower,ymax=upper),width=0.2) +
xlab("") +
ylab("Predictor importance\n(Leave-one-out C-statistic decrease)") +
labs(fill=expression(Overlap~with~ML4H[EN-COX])) +
theme_classic() + 
theme(axis.text.x = element_text(angle=90,hjust=0.95,vjust=0.5),text = element_text(size=16),
     legend.position=c(0.8,0.6)) +
# scale_fill_discrete(drop=FALSE) +
scale_fill_brewer(palette = 'Set2',drop=FALSE) +
scale_y_continuous(expand=expansion(mult = c(0,0.2))) +
scale_x_discrete(breaks=xgcoxLeave1OutLabel[1:20]$abbrev,
                labels=c('Age',expression(GPS[CAD]),'Sex','SBP','ApoB','Testosterone',
                         'HbA1C','Cystatin C','HDL-c','Father heart dz','Current smoker',
                         'Sibling heart dz','ApoA','Lp(a)','CRP','Fair health rating',
                         'Hypertension','Poor health rating','Height','LDL-c'))
#the drop=FALSE allows all categories to remain even though I subsetting the datatable

xgCoxFeatPlot

# ggsave('coxnet51_leave1out.pdf',g2,width=10,height=8)

coxnet_perm_imp_clean_corrplot <- coxnet_perm_imp_clean[field %in% coxnet_feature_list_annot_clean$field]

corrplot_dt <- merge(coxnet_feature_list_annot_clean,
                     coxnet_perm_imp_clean_corrplot[,c('field','V1')],
                    by='field',all=TRUE)

corrplot_dt[is.na(V1),V1:=0]

corrplot_dt <- corrplot_dt[order(-V1)]
corrplot_dt <- corrplot_dt[order(category)]

fullfeat_dev_corrplot <- setcolorder(fullfeat_devUnscaled,c('index',corrplot_dt$field))


names(fullfeat_dev_corrplot)[-1] <- corrplot_dt$abbrev

M <- cor(fullfeat_dev_corrplot[,-c('index')])

options(repr.plot.width=14,repr.plot.height=14)
corrplot(M,type='upper',order='original',col=brewer.pal(n=8,name='RdYlBu'),
         tl.col='black')

# pdf(height=16,width=16,file='/medpop/esp2/sagrawal/mi_mlprediction/2020.11.30_marcus/mlpaperplots/efigure6.pdf')

# corrplot(M,type='upper',order='original',col=brewer.pal(n=8,name='RdYlBu'),
#          tl.col='black')

# dev.off()

fullfeat_dev <- fread("coxnet__feature_matrix__development_0_0011765691116882482__30Nov_2020.txt")
fullfeat_ho <- fread("coxnet__feature_matrix__holdout_0_0011765691116882482__30Nov_2020.txt")

dev_merged <- merge(fullfeat_dev,masterdf[,c('eid','CAD_incident_disease','CAD_fu_days')],
                   by.x='index',by.y='eid')

simplecox_cph <- coxph(Surv(CAD_fu_days,CAD_incident_disease) ~ ., data=dev_merged[,-c('index')])
summary(simplecox_cph)

simplecox_cph_20 <- coxph(Surv(CAD_fu_days,CAD_incident_disease) ~ d_age +
                          d_prsadjscore + c_sex + d_biochemistry_30760 +
                          d_systolic_bp + d_biochemistry_30750 +
                          d_biochemistry_30780 + d_biochemistry_30850 +
                          d_hip + d_biochemistry_30640 +
                          d_waist + father_1 + d_biochemistry_30720 +
                          d_lipoprotein_a + c_selfreport_icd10_20002_1065 +
                          c_overall_health_3 + d_biochemistry_30140 +
                          sibling_1 + d_PC3 + c_lipidlowering, data=dev_merged)
summary(simplecox_cph_20)

fullfeat_dev <- fread("coxnet__feature_matrix__development_0_0011765691116882482__30Nov_2020.txt")
fullfeat_ho <- fread("coxnet__feature_matrix__holdout_0_0011765691116882482__30Nov_2020.txt")

masterdf_dev1_pred <- masterdf[subgroup_indicator == 1]
masterdf_ho1_pred <- masterdf[subgroup_indicator == 2]
masterdf_devho1_pred <- masterdf[subgroup_indicator != 0]

tfunc <- function(vec){
        vec2 <- log(-1*log(1-vec))
        return(vec2)
    }

undo_func <- function(vec){
    vec2 <- 1 - (exp(-1*exp(vec)))
    return(vec2)
}

EHJ2019Cox <- coxph(Surv(CAD_fu_days,CAD_incident_disease) ~ ageG2 + ageG3 + ageG4 + ageG5 + ageG6 + ageG7,
                    data=masterdf_dev1_pred[sex == 'Male'])

EHJ2019Cox_10yr <- coxph(Surv(CAD_fu_days10yr,CAD_tenyearoutcome) ~ ageG2 + ageG3 + ageG4 + ageG5 + ageG6 + ageG7,
                    data=masterdf_dev1_pred[sex == 'Male'])

EHJ2019CoxF <- coxph(Surv(CAD_fu_days,CAD_incident_disease) ~ ageG2 + ageG3 + ageG4 + ageG5 + ageG6 + ageG7,
                    data=masterdf_dev1_pred[sex == 'Female'])

EHJ2019Cox_10yrF <- coxph(Surv(CAD_fu_days10yr,CAD_tenyearoutcome) ~ ageG2 + ageG3 + ageG4 + ageG5 + ageG6 + ageG7,
                    data=masterdf_dev1_pred[sex == 'Female'])

dummyDT <- data.table(ageG1 = c(1,rep(0,6)),
                     ageG2 = c(0,1,rep(0,5)),
                      ageG3 = c(0,0,1,rep(0,4)),
                      ageG4 = c(0,0,0,1,0,0,0),
                      ageG5 = c(rep(0,4),1,0,0),
                      ageG6 = c(rep(0,5),1,0),
                      ageG7 = c(rep(0,6),1))

MalePred <- absolute_risk(EHJ2019Cox,dummyDT)

FemalePred <- absolute_risk(EHJ2019CoxF,dummyDT)

ext_cal_mod <- function(dt_in,predcol,sortcol,extpred){
    intpred <- 
    tapply(dt_in[[predcol]],dt_in[[sortcol]],mean)
    
    dt <- as.data.table(cbind(intpred,extpred))
    names(dt) <- c("intpred","extpred")
    
    lm1 <- lm(tfunc(extpred) ~ tfunc(intpred),data = dt)
    return(lm1)
}

##Note that the requirement to make sure tenyearASCVD or qrisk3 are not missing
##omits 9 entries for former and 6-7% entries for latter.

PCE_recal_male <- ext_cal_mod(masterdf_dev1_pred[sex=='Male' & !is.na(tenyearASCVD)],"tenyearASCVD","age_group",MalePred)
PCE_recal_female <- ext_cal_mod(masterdf_dev1_pred[sex=='Female' & !is.na(tenyearASCVD)],"tenyearASCVD","age_group",FemalePred)

QRISK3_recal_male <- ext_cal_mod(masterdf_dev1_pred[sex=='Male' & !is.na(qrisk3)],"qrisk3","age_group",MalePred)
QRISK3_recal_female <- ext_cal_mod(masterdf_dev1_pred[sex=='Female' & !is.na(qrisk3)],"qrisk3","age_group",FemalePred)

fram_recal_male <- ext_cal_mod(masterdf_dev1_pred[sex=='Male' & !is.na(framrisk)],"framrisk","age_group",MalePred)
fram_recal_female <- ext_cal_mod(masterdf_dev1_pred[sex=='Female' & !is.na(framrisk)],"framrisk","age_group",FemalePred)

masterdf_ho1_pred[sex=='Male',tenyearASCVD_recal:=undo_func((tfunc(tenyearASCVD) * PCE_recal_male$coefficients[["tfunc(intpred)"]]) +
    PCE_recal_male$coefficients[["(Intercept)"]])]

masterdf_ho1_pred[sex=='Female',tenyearASCVD_recal:=undo_func((tfunc(tenyearASCVD) * PCE_recal_female$coefficients[["tfunc(intpred)"]]) +
    PCE_recal_female$coefficients[["(Intercept)"]])]

masterdf_dev1_pred[sex=='Male',tenyearASCVD_recal:=undo_func((tfunc(tenyearASCVD) * PCE_recal_male$coefficients[["tfunc(intpred)"]]) +
    PCE_recal_male$coefficients[["(Intercept)"]])]

masterdf_dev1_pred[sex=='Female',tenyearASCVD_recal:=undo_func((tfunc(tenyearASCVD) * PCE_recal_female$coefficients[["tfunc(intpred)"]]) +
    PCE_recal_female$coefficients[["(Intercept)"]])]

masterdf_devho1_pred[sex=='Male',tenyearASCVD_recal:=undo_func((tfunc(tenyearASCVD) * PCE_recal_male$coefficients[["tfunc(intpred)"]]) +
    PCE_recal_male$coefficients[["(Intercept)"]])]

masterdf_devho1_pred[sex=='Female',tenyearASCVD_recal:=undo_func((tfunc(tenyearASCVD) * PCE_recal_female$coefficients[["tfunc(intpred)"]]) +
    PCE_recal_female$coefficients[["(Intercept)"]])]

masterdf_ho1_pred[sex=='Male',qrisk3_recal:=undo_func((tfunc(qrisk3) * QRISK3_recal_male$coefficients[["tfunc(intpred)"]]) +
    QRISK3_recal_male$coefficients[["(Intercept)"]])]

masterdf_ho1_pred[sex=='Female',qrisk3_recal:=undo_func((tfunc(qrisk3) * QRISK3_recal_female$coefficients[["tfunc(intpred)"]]) +
    QRISK3_recal_female$coefficients[["(Intercept)"]])]

masterdf_dev1_pred[sex=='Male',qrisk3_recal:=undo_func((tfunc(qrisk3) * QRISK3_recal_male$coefficients[["tfunc(intpred)"]]) +
    QRISK3_recal_male$coefficients[["(Intercept)"]])]

masterdf_dev1_pred[sex=='Female',qrisk3_recal:=undo_func((tfunc(qrisk3) * QRISK3_recal_female$coefficients[["tfunc(intpred)"]]) +
    QRISK3_recal_female$coefficients[["(Intercept)"]])]

masterdf_devho1_pred[sex=='Male',qrisk3_recal:=undo_func((tfunc(qrisk3) * QRISK3_recal_male$coefficients[["tfunc(intpred)"]]) +
    QRISK3_recal_male$coefficients[["(Intercept)"]])]

masterdf_devho1_pred[sex=='Female',qrisk3_recal:=undo_func((tfunc(qrisk3) * QRISK3_recal_female$coefficients[["tfunc(intpred)"]]) +
    QRISK3_recal_female$coefficients[["(Intercept)"]])]

masterdf_ho1_pred[sex=='Male',framrisk_recal:=undo_func((tfunc(framrisk) * fram_recal_male$coefficients[["tfunc(intpred)"]]) +
    fram_recal_male$coefficients[["(Intercept)"]])]

masterdf_ho1_pred[sex=='Female',framrisk_recal:=undo_func((tfunc(framrisk) * fram_recal_female$coefficients[["tfunc(intpred)"]]) +
    fram_recal_female$coefficients[["(Intercept)"]])]

masterdf_dev1_pred[sex=='Male',framrisk_recal:=undo_func((tfunc(framrisk) * fram_recal_male$coefficients[["tfunc(intpred)"]]) +
    fram_recal_male$coefficients[["(Intercept)"]])]

masterdf_dev1_pred[sex=='Female',framrisk_recal:=undo_func((tfunc(framrisk) * fram_recal_female$coefficients[["tfunc(intpred)"]]) +
    fram_recal_female$coefficients[["(Intercept)"]])]

masterdf_devho1_pred[sex=='Male',framrisk_recal:=undo_func((tfunc(framrisk) * fram_recal_male$coefficients[["tfunc(intpred)"]]) +
    fram_recal_male$coefficients[["(Intercept)"]])]

masterdf_devho1_pred[sex=='Female',framrisk_recal:=undo_func((tfunc(framrisk) * fram_recal_female$coefficients[["tfunc(intpred)"]]) +
    fram_recal_female$coefficients[["(Intercept)"]])]

coxnet_abs_id <- fread("coxnet__absolute_risk_with_ids__development_model_0_0011765691116882482__30Nov_2020.txt")
coxnet_abs_ho_id <- fread("coxnet__absolute_risk_with_ids__holdout_model_0_0011765691116882482__30Nov_2020.txt")

coxnet_abs_ho_id$coxnet51risk <- coxnet_abs_ho_id$abs_risk_t10
coxnet_abs_id$coxnet51risk <- coxnet_abs_id$abs_risk_t10

masterdf_ho1_pred1 <- merge(masterdf_ho1_pred,coxnet_abs_ho_id[,c('ukbid','coxnet51risk')],
                           by.x='eid',by.y='ukbid')

masterdf_dev1_pred1 <- merge(masterdf_dev1_pred,coxnet_abs_id[,c('ukbid','coxnet51risk')],
                           by.x='eid',by.y='ukbid')

simplecox_feat1 <- fread("coxph__feature_list_0_0011765691116882482__30Nov_2020.txt")
simplecox_feat1 <- simplecox_feat1[-1]

all.equal(simplecox_feat1$V1,names(fullfeat_dev[,-c('index')]))

featurenames_simplecox <- names(fullfeat_dev[,-c('index')])

simplecox_newlp_dev <- fullfeat_dev[,Calc := as.matrix(fullfeat_dev[,..featurenames_simplecox])%*%simplecox_feat1$V2]
simplecox_newlp_ho <- fullfeat_ho[,Calc := as.matrix(fullfeat_ho[,..featurenames_simplecox])%*%simplecox_feat1$V2]

coxnet_abs_id$simplecox_newlp <- simplecox_newlp_dev$Calc
coxnet_abs_ho_id$simplecox_newlp <- simplecox_newlp_ho$Calc

cph_simplecox_dev <- coxph(Surv(time,O) ~ simplecox_newlp,data=coxnet_abs_id)
cumhaz5 <- basehaz(cph_simplecox_dev,centered = TRUE)
basesurv10_5 <- exp(-1*cumhaz5$hazard[which.min(abs(cumhaz5$time - 3652))])
coxnet_abs_ho_id$simpcox51risk <- 1 - basesurv10_5^exp((coxnet_abs_ho_id$simplecox_newlp - mean(coxnet_abs_id$simplecox_newlp)))
coxnet_abs_id$simpcox51risk <- 1 - basesurv10_5^exp((coxnet_abs_id$simplecox_newlp - mean(coxnet_abs_id$simplecox_newlp)))

coxnet_abs_ho_id$simpcox20risk <- absolute_risk(simplecox_cph_20,data=fullfeat_ho)
coxnet_abs_id$simpcox20risk <- absolute_risk(simplecox_cph_20,data=fullfeat_dev)

masterdf_ho1_pred2 <- merge(masterdf_ho1_pred1,coxnet_abs_ho_id[,c('ukbid','simpcox51risk','simpcox20risk')],
                           by.x='eid',by.y='ukbid')

masterdf_dev1_pred2 <- merge(masterdf_dev1_pred1,coxnet_abs_id[,c('ukbid','simpcox51risk','simpcox20risk')],
                           by.x='eid',by.y='ukbid')

xgcoxAbsRiskHo_clean <- xgcoxAbsRiskHo[,c('ukbid','abs_risk_t10')]
names(xgcoxAbsRiskHo_clean)[2] <- 'xgcoxrisk'

masterdf_ho1_pred2 <- merge(masterdf_ho1_pred2,xgcoxAbsRiskHo_clean,by.x='eid',by.y='ukbid')

dtforcorrEFig9 <- masterdf_ho1_pred2[,c('coxnet51risk','simpcox51risk','xgcoxrisk',
                                   'framrisk_recal','tenyearASCVD_recal','qrisk3_recal')]

names(dtforcorrEFig9) <- c('ML4HEN-COX','SimpleCox51','XGBoost','FRS','PCE','QRISK3')

M4 <- cor(dtforcorrEFig9,use = 'complete.obs')
M4

# pdf(height=8,width=8,file='/medpop/esp2/sagrawal/mi_mlprediction/2020.11.30_marcus/mlpaperplots/efigure9.pdf')
options(repr.plot.width=8,repr.plot.height=8)
corrplot(M4,type='upper',order='original',col=brewer.pal(n=10,name='Paired'),
         tl.col='black',cl.lim=c(0.7,1),is.corr=FALSE)

# dev.off()

options(repr.plot.width=14,repr.plot.height=7)

coxnet51_cal_dev <- calibration_plot(data=masterdf_dev1_pred2,
                predictioncol='coxnet51risk',
                eventcol='CAD_tenyearoutcome')

coxnet51_cal_ho <- calibration_plot(data=masterdf_ho1_pred2,
                predictioncol='coxnet51risk',
                eventcol='CAD_tenyearoutcome')

coxnet_cal <- coxnet51_cal_dev + coxnet51_cal_ho + plot_layout(nrow=1)
coxnet_cal

# ggsave('/medpop/esp2/sagrawal/mi_mlprediction/2020.11.30_marcus/mlpaperplots/efigure4_alt.pdf',
#        coxnet_cal,width=14,height=7)

options(repr.plot.width=14,repr.plot.height=14)

xgboost_cal_ho <- calibration_plot(data=masterdf_ho1_pred2,
                predictioncol='xgcoxrisk',
                eventcol='CAD_tenyearoutcome')

simpcox51_cal_ho <- calibration_plot(data=masterdf_ho1_pred2,
                predictioncol='simpcox51risk',
                eventcol='CAD_tenyearoutcome')

simpcox20_cal_ho <- calibration_plot(data=masterdf_ho1_pred2,
                predictioncol='simpcox20risk',
                eventcol='CAD_tenyearoutcome')

SnAnalysis_cal <- xgboost_cal_ho + simpcox51_cal_ho + simpcox20_cal_ho + plot_layout(nrow=2,ncol=2)
SnAnalysis_cal

# ggsave('/medpop/esp2/sagrawal/mi_mlprediction/2020.11.30_marcus/mlpaperplots/efigure5_alt.pdf',
#        SnAnalysis_cal,width=14,height=14)

options(repr.plot.width=14,repr.plot.height=7)

pce_cal_dev <- calibration_plot(data=masterdf_dev1_pred[!is.na(tenyearASCVD)],
                predictioncol='tenyearASCVD',
                eventcol='CAD_tenyearoutcome',
                manualpvalue='p-value < 0.001')

pce_recal_cal_dev <- calibration_plot(data=masterdf_dev1_pred[!is.na(tenyearASCVD)],
                predictioncol='tenyearASCVD_recal',
                eventcol='CAD_tenyearoutcome')

pce_plots_dev <- pce_cal_dev + pce_recal_cal_dev 
pce_plots_dev

# ggsave('/medpop/esp2/sagrawal/mi_mlprediction/2020.11.30_marcus/mlpaperplots/pce_calplots_dev.pdf',
#        pce_plots_dev,width=14,height=7)

options(repr.plot.width=14,repr.plot.height=7)

pce_cal_ho <- calibration_plot(data=masterdf_ho1_pred[!is.na(tenyearASCVD)],
                predictioncol='tenyearASCVD',
                eventcol='CAD_tenyearoutcome',
                manualpvalue='p-value < 0.001')

pce_recal_cal_ho <- calibration_plot(data=masterdf_ho1_pred[!is.na(tenyearASCVD)],
                predictioncol='tenyearASCVD_recal',
                eventcol='CAD_tenyearoutcome')

pce_plots_ho <- pce_cal_ho + pce_recal_cal_ho 
pce_plots_ho

# ggsave('/medpop/esp2/sagrawal/mi_mlprediction/2020.11.30_marcus/mlpaperplots/pce_calplots.pdf',
#        pce_plots,width=14,height=7)

options(repr.plot.width=14,repr.plot.height=7)

qrisk3_cal_dev <- calibration_plot(data=masterdf_dev1_pred[!is.na(qrisk3)],
                predictioncol='qrisk3',
                eventcol='CAD_tenyearoutcome',
                manualpvalue='p-value < 0.001')

qrisk3_recal_cal_dev <- calibration_plot(data=masterdf_dev1_pred[!is.na(qrisk3_recal)],
                predictioncol='qrisk3_recal',
                eventcol='CAD_tenyearoutcome')

qrisk3_plots_dev <- qrisk3_cal_dev + qrisk3_recal_cal_dev
qrisk3_plots_dev

# ggsave('/medpop/esp2/sagrawal/mi_mlprediction/2020.11.30_marcus/mlpaperplots/qrisk3_calplots_dev.pdf',
#        qrisk3_plots_dev,width=14,height=7)

options(repr.plot.width=14,repr.plot.height=7)

qrisk3_cal_ho <- calibration_plot(data=masterdf_ho1_pred[!is.na(qrisk3)],
                predictioncol='qrisk3',
                eventcol='CAD_tenyearoutcome',
                manualpvalue='p-value < 0.001')

qrisk3_recal_cal_ho <- calibration_plot(data=masterdf_ho1_pred[!is.na(qrisk3_recal)],
                predictioncol='qrisk3_recal',
                eventcol='CAD_tenyearoutcome')

qrisk3_plots_ho <- qrisk3_cal_ho + qrisk3_recal_cal_ho
qrisk3_plots_ho

# ggsave('/medpop/esp2/sagrawal/mi_mlprediction/2020.11.30_marcus/mlpaperplots/qrisk3_calplots.pdf',
#        qrisk3_plots,width=14,height=7)

options(repr.plot.width=14,repr.plot.height=7)

framrisk_cal_dev <- calibration_plot(data=masterdf_dev1_pred[!is.na(framrisk)],
                predictioncol='framrisk',
                eventcol='CAD_tenyearoutcome',
                manualpvalue='p-value < 0.001')

framrisk_recal_cal_dev <- calibration_plot(data=masterdf_dev1_pred[!is.na(framrisk_recal)],
                predictioncol='framrisk_recal',
                eventcol='CAD_tenyearoutcome')

fram_plots_dev <- framrisk_cal_dev + framrisk_recal_cal_dev
fram_plots_dev

# ggsave('/medpop/esp2/sagrawal/mi_mlprediction/2020.11.30_marcus/mlpaperplots/framrisk_calplots_dev.pdf',
#        fram_plots_dev,width=14,height=7)

options(repr.plot.width=14,repr.plot.height=7)

framrisk_cal_ho <- calibration_plot(data=masterdf_ho1_pred[!is.na(framrisk)],
                predictioncol='framrisk',
                eventcol='CAD_tenyearoutcome',
                manualpvalue='p-value < 0.001')

framrisk_recal_cal_ho <- calibration_plot(data=masterdf_ho1_pred[!is.na(framrisk_recal)],
                predictioncol='framrisk_recal',
                eventcol='CAD_tenyearoutcome')

fram_plots_ho <- framrisk_cal_ho + framrisk_recal_cal_ho
fram_plots_ho

# ggsave('/medpop/esp2/sagrawal/mi_mlprediction/2020.11.30_marcus/mlpaperplots/framrisk_calplots.pdf',
#        fram_plots,width=14,height=7)

#Purpose of this is simply to correctly calculate DeLong p-value between ML4HEN-COX and QRISK3
masterdf_ho1_pred3 <- masterdf_ho1_pred2[!is.na(qrisk3_recal)]

rocobj_coxnet51 <- roc(masterdf_ho1_pred2$CAD_tenyearoutcome,masterdf_ho1_pred2$coxnet51risk)
rocobj_coxnet51_2 <- roc(masterdf_ho1_pred3$CAD_tenyearoutcome,masterdf_ho1_pred3$coxnet51risk)
rocobj_simpcox51risk <- roc(masterdf_ho1_pred2$CAD_tenyearoutcome,masterdf_ho1_pred2$simpcox51risk)
rocobj_simpcox20risk <- roc(masterdf_ho1_pred2$CAD_tenyearoutcome,masterdf_ho1_pred2$simpcox20risk)
rocobj_framrisk <- roc(masterdf_ho1_pred2[!is.na(framrisk_recal)]$CAD_tenyearoutcome,masterdf_ho1_pred2[!is.na(framrisk_recal)]$framrisk_recal)
rocobj_tenyearASCVD <- roc(masterdf_ho1_pred2[!is.na(tenyearASCVD_recal)]$CAD_tenyearoutcome,masterdf_ho1_pred2[!is.na(tenyearASCVD_recal)]$tenyearASCVD_recal)
rocobj_qrisk3 <- roc(masterdf_ho1_pred3$CAD_tenyearoutcome,masterdf_ho1_pred3$qrisk3_recal)
rocobj_xgcoxrisk <- roc(masterdf_ho1_pred2$CAD_tenyearoutcome,masterdf_ho1_pred2$xgcoxrisk)

roc.test(rocobj_coxnet51,rocobj_framrisk,method='delong')
roc.test(rocobj_coxnet51,rocobj_tenyearASCVD,method='delong')
roc.test(rocobj_coxnet51_2,rocobj_qrisk3,method='delong')
roc.test(rocobj_coxnet51,rocobj_xgcoxrisk,method='delong')
roc.test(rocobj_coxnet51,rocobj_simpcox51risk,method='delong')
roc.test(rocobj_coxnet51,rocobj_simpcox20risk,method='delong')

ci(rocobj_qrisk3,method='bootstrap',boot.n=1000,progress='text')

rocobj_coxnet51_male <- roc(masterdf_ho1_pred2[sex=='Male']$CAD_tenyearoutcome,masterdf_ho1_pred2[sex=='Male']$coxnet51risk)
rocobj_coxnet51_male_2 <- roc(masterdf_ho1_pred3[sex=='Male']$CAD_tenyearoutcome,masterdf_ho1_pred3[sex=='Male']$coxnet51risk)
rocobj_simpcox51risk_male <- roc(masterdf_ho1_pred2[sex=='Male']$CAD_tenyearoutcome,masterdf_ho1_pred2[sex=='Male']$simpcox51risk)
rocobj_simpcox20risk_male <- roc(masterdf_ho1_pred2[sex=='Male']$CAD_tenyearoutcome,masterdf_ho1_pred2[sex=='Male']$simpcox20risk)
rocobj_framrisk_male <- roc(masterdf_ho1_pred2[sex=='Male' & !is.na(framrisk_recal)]$CAD_tenyearoutcome,masterdf_ho1_pred2[sex=='Male' & !is.na(framrisk_recal)]$framrisk_recal)
rocobj_tenyearASCVD_male <- roc(masterdf_ho1_pred2[sex=='Male' & !is.na(tenyearASCVD_recal)]$CAD_tenyearoutcome,masterdf_ho1_pred2[sex=='Male' & !is.na(tenyearASCVD_recal)]$tenyearASCVD_recal)
rocobj_qrisk3_male <- roc(masterdf_ho1_pred3[sex=='Male']$CAD_tenyearoutcome,masterdf_ho1_pred3[sex=='Male']$qrisk3_recal)
rocobj_xgcoxrisk_male <- roc(masterdf_ho1_pred2[sex=='Male']$CAD_tenyearoutcome,masterdf_ho1_pred2[sex=='Male']$xgcoxrisk)

roc.test(rocobj_coxnet51_male,rocobj_framrisk_male,method='delong')
roc.test(rocobj_coxnet51_male,rocobj_tenyearASCVD_male,method='delong')
roc.test(rocobj_coxnet51_male_2,rocobj_qrisk3_male,method='delong')
roc.test(rocobj_coxnet51_male,rocobj_xgcoxrisk_male,method='delong')
roc.test(rocobj_coxnet51_male,rocobj_simpcox51risk_male,method='delong')
roc.test(rocobj_coxnet51_male,rocobj_simpcox20risk_male,method='delong')

rocobj_coxnet51_female <- roc(masterdf_ho1_pred2[sex=='Female']$CAD_tenyearoutcome,masterdf_ho1_pred2[sex=='Female']$coxnet51risk)
rocobj_coxnet51_female_2 <- roc(masterdf_ho1_pred3[sex=='Female']$CAD_tenyearoutcome,masterdf_ho1_pred3[sex=='Female']$coxnet51risk)
rocobj_simpcox51risk_female <- roc(masterdf_ho1_pred2[sex=='Female']$CAD_tenyearoutcome,masterdf_ho1_pred2[sex=='Female']$simpcox51risk)
rocobj_simpcox20risk_female <- roc(masterdf_ho1_pred2[sex=='Female']$CAD_tenyearoutcome,masterdf_ho1_pred2[sex=='Female']$simpcox20risk)
rocobj_framrisk_female <- roc(masterdf_ho1_pred2[sex=='Female' & !is.na(framrisk_recal)]$CAD_tenyearoutcome,masterdf_ho1_pred2[sex=='Female' & !is.na(framrisk_recal)]$framrisk_recal)
rocobj_tenyearASCVD_female <- roc(masterdf_ho1_pred2[sex=='Female' & !is.na(tenyearASCVD_recal)]$CAD_tenyearoutcome,masterdf_ho1_pred2[sex=='Female' & !is.na(tenyearASCVD_recal)]$tenyearASCVD_recal)
rocobj_qrisk3_female <- roc(masterdf_ho1_pred3[sex=='Female']$CAD_tenyearoutcome,masterdf_ho1_pred3[sex=='Female']$qrisk3_recal)
rocobj_xgcoxrisk_female <- roc(masterdf_ho1_pred2[sex=='Female']$CAD_tenyearoutcome,masterdf_ho1_pred2[sex=='Female']$xgcoxrisk)

roc.test(rocobj_coxnet51_female,rocobj_framrisk_female,method='delong')
roc.test(rocobj_coxnet51_female,rocobj_tenyearASCVD_female,method='delong')
roc.test(rocobj_coxnet51_female_2,rocobj_qrisk3_female,method='delong')

roc.test(rocobj_coxnet51_female,rocobj_xgcoxrisk_female,method='delong')
roc.test(rocobj_coxnet51_female,rocobj_simpcox51risk_female,method='delong')
roc.test(rocobj_coxnet51_female,rocobj_simpcox20risk_female,method='delong')

rocobj_coxnet51_under55 <- roc(masterdf_ho1_pred2[age<55]$CAD_tenyearoutcome,masterdf_ho1_pred2[age<55]$coxnet51risk)
rocobj_coxnet51_under55_2 <- roc(masterdf_ho1_pred3[age<55]$CAD_tenyearoutcome,masterdf_ho1_pred3[age<55]$coxnet51risk)
rocobj_simpcox51risk_under55 <- roc(masterdf_ho1_pred2[age<55]$CAD_tenyearoutcome,masterdf_ho1_pred2[age<55]$simpcox51risk)
rocobj_simpcox20risk_under55 <- roc(masterdf_ho1_pred2[age<55]$CAD_tenyearoutcome,masterdf_ho1_pred2[age<55]$simpcox20risk)
rocobj_framrisk_under55 <- roc(masterdf_ho1_pred2[age<55 & !is.na(framrisk_recal)]$CAD_tenyearoutcome,masterdf_ho1_pred2[age<55 & !is.na(framrisk_recal)]$framrisk_recal)
rocobj_tenyearASCVD_under55 <- roc(masterdf_ho1_pred2[age<55 & !is.na(tenyearASCVD_recal)]$CAD_tenyearoutcome,masterdf_ho1_pred2[age<55 & !is.na(tenyearASCVD_recal)]$tenyearASCVD_recal)
rocobj_qrisk3_under55 <- roc(masterdf_ho1_pred3[age<55]$CAD_tenyearoutcome,masterdf_ho1_pred3[age<55]$qrisk3_recal)
rocobj_xgcoxrisk_under55 <- roc(masterdf_ho1_pred2[age<55]$CAD_tenyearoutcome,masterdf_ho1_pred2[age<55]$xgcoxrisk)

roc.test(rocobj_coxnet51_under55,rocobj_framrisk_under55,method='delong')
roc.test(rocobj_coxnet51_under55,rocobj_tenyearASCVD_under55,method='delong')
roc.test(rocobj_coxnet51_under55_2,rocobj_qrisk3_under55,method='delong')

roc.test(rocobj_coxnet51_under55,rocobj_xgcoxrisk_under55,method='delong')
roc.test(rocobj_coxnet51_under55,rocobj_simpcox51risk_under55,method='delong')
roc.test(rocobj_coxnet51_under55,rocobj_simpcox20risk_under55,method='delong')

rocobj_coxnet51_over55 <- roc(masterdf_ho1_pred2[age>=55]$CAD_tenyearoutcome,masterdf_ho1_pred2[age>=55]$coxnet51risk)
rocobj_coxnet51_over55_2 <- roc(masterdf_ho1_pred3[age>=55]$CAD_tenyearoutcome,masterdf_ho1_pred3[age>=55]$coxnet51risk)

rocobj_simpcox51risk_over55 <- roc(masterdf_ho1_pred2[age>=55]$CAD_tenyearoutcome,masterdf_ho1_pred2[age>=55]$simpcox51risk)
rocobj_simpcox20risk_over55 <- roc(masterdf_ho1_pred2[age>=55]$CAD_tenyearoutcome,masterdf_ho1_pred2[age>=55]$simpcox20risk)
rocobj_framrisk_over55 <- roc(masterdf_ho1_pred2[age>=55 & !is.na(framrisk_recal)]$CAD_tenyearoutcome,masterdf_ho1_pred2[age>=55 & !is.na(framrisk_recal)]$framrisk_recal)
rocobj_tenyearASCVD_over55 <- roc(masterdf_ho1_pred2[age>=55 & !is.na(tenyearASCVD_recal)]$CAD_tenyearoutcome,masterdf_ho1_pred2[age>=55 & !is.na(tenyearASCVD_recal)]$tenyearASCVD_recal)
rocobj_qrisk3_over55 <- roc(masterdf_ho1_pred3[age>=55]$CAD_tenyearoutcome,masterdf_ho1_pred3[age>=55]$qrisk3_recal)
rocobj_xgcoxrisk_over55 <- roc(masterdf_ho1_pred2[age>=55]$CAD_tenyearoutcome,masterdf_ho1_pred2[age>=55]$xgcoxrisk)

roc.test(rocobj_coxnet51_over55,rocobj_framrisk_over55,method='delong')
roc.test(rocobj_coxnet51_over55,rocobj_tenyearASCVD_over55,method='delong')
roc.test(rocobj_coxnet51_over55_2,rocobj_qrisk3_over55,method='delong')

roc.test(rocobj_coxnet51_over55,rocobj_xgcoxrisk_over55,method='delong')
roc.test(rocobj_coxnet51_over55,rocobj_simpcox51risk_over55,method='delong')
roc.test(rocobj_coxnet51_over55,rocobj_simpcox20risk_over55,method='delong')

rocobj_tenyearASCVD_1 <- roc(masterdf_ho1_pred2[!is.na(tenyearASCVD_recal)]$CAD_tenyearoutcome,masterdf_ho1_pred2[!is.na(tenyearASCVD_recal)]$tenyearASCVD_recal)
rocobj_qrisk3_1 <- roc(masterdf_ho1_pred2[!is.na(qrisk3_recal)]$CAD_tenyearoutcome,masterdf_ho1_pred2[!is.na(qrisk3_recal)]$qrisk3_recal)

rocobj_tenyearASCVD_1
rocobj_qrisk3_1

rocobj_tenyearASCVD_1 <- roc(masterdf_ho1_pred2[!is.na(tenyearASCVD)]$CAD_tenyearoutcome,masterdf_ho1_pred2[!is.na(tenyearASCVD)]$tenyearASCVD_recal)
rocobj_qrisk3_1 <- roc(masterdf_ho1_pred2[!is.na(qrisk3)]$CAD_tenyearoutcome,masterdf_ho1_pred2[!is.na(qrisk3)]$qrisk3_recal)

rocobj_tenyearASCVD_1
rocobj_qrisk3_1

rocobj_tenyearASCVD_1 <- roc(masterdf_ho1_pred2[!is.na(tenyearASCVD)]$CVD_tenyearoutcome,masterdf_ho1_pred2[!is.na(tenyearASCVD)]$tenyearASCVD_recal)
rocobj_qrisk3_1 <- roc(masterdf_ho1_pred2[!is.na(qrisk3)]$CVD_tenyearoutcome,masterdf_ho1_pred2[!is.na(qrisk3)]$qrisk3_recal)
rocobj_coxnet51_1 <- roc(masterdf_ho1_pred2$CVD_tenyearoutcome,masterdf_ho1_pred2$coxnet51risk)

rocobj_tenyearASCVD_1
rocobj_qrisk3_1
rocobj_coxnet51_1

ci(rocobj_tenyearASCVD_1,method='bootstrap',boot.n=1000,progress='text')
ci(rocobj_qrisk3_1,method='bootstrap',boot.n=1000,progress='text')
ci(rocobj_coxnet51_1,method='bootstrap',boot.n=1000,progress='text')

reclass(data=masterdf_ho1_pred2,
                 cOutcome="CAD_tenyearoutcome",
                predrisk1=masterdf_ho1_pred2$framrisk_recal,
                predrisk2=masterdf_ho1_pred2$coxnet51risk,
                cutoff=c(0,0.025,1))

reclass(data=masterdf_ho1_pred2,
                 cOutcome="CAD_tenyearoutcome",
                predrisk1=masterdf_ho1_pred2$tenyearASCVD_recal,
                predrisk2=masterdf_ho1_pred2$coxnet51risk,
                cutoff=c(0,0.025,1))

reclass(data=masterdf_ho1_pred2[!is.na(qrisk3_recal)],
                 cOutcome="CAD_tenyearoutcome",
                predrisk1=masterdf_ho1_pred2[!is.na(qrisk3_recal)]$qrisk3_recal,
                predrisk2=masterdf_ho1_pred2[!is.na(qrisk3_recal)]$coxnet51risk,
                cutoff=c(0,0.025,1))


reclass(data=masterdf_ho1_pred2,
                 cOutcome="CAD_tenyearoutcome",
                predrisk1=masterdf_ho1_pred2$framrisk_recal,
                predrisk2=masterdf_ho1_pred2$coxnet51risk,
                cutoff=c(0,0.05,1))

reclass(data=masterdf_ho1_pred2,
                 cOutcome="CAD_tenyearoutcome",
                predrisk1=masterdf_ho1_pred2$tenyearASCVD_recal,
                predrisk2=masterdf_ho1_pred2$coxnet51risk,
                cutoff=c(0,0.05,1))

reclass(data=masterdf_ho1_pred2[!is.na(qrisk3_recal)],
                 cOutcome="CAD_tenyearoutcome",
                predrisk1=masterdf_ho1_pred2[!is.na(qrisk3_recal)]$qrisk3_recal,
                predrisk2=masterdf_ho1_pred2[!is.na(qrisk3_recal)]$coxnet51risk,
                cutoff=c(0,0.05,1))

reclass(data=masterdf_ho1_pred2,
                 cOutcome="CAD_tenyearoutcome",
                predrisk1=masterdf_ho1_pred2$framrisk_recal,
                predrisk2=masterdf_ho1_pred2$simpcox51risk,
                cutoff=c(0,0.025,1))

reclass(data=masterdf_ho1_pred2,
                 cOutcome="CAD_tenyearoutcome",
                predrisk1=masterdf_ho1_pred2$tenyearASCVD_recal,
                predrisk2=masterdf_ho1_pred2$simpcox51risk,
                cutoff=c(0,0.025,1))

reclass(data=masterdf_ho1_pred2[!is.na(qrisk3_recal)],
                 cOutcome="CAD_tenyearoutcome",
                predrisk1=masterdf_ho1_pred2[!is.na(qrisk3_recal)]$qrisk3_recal,
                predrisk2=masterdf_ho1_pred2[!is.na(qrisk3_recal)]$simpcox51risk,
                cutoff=c(0,0.025,1))

reclass(data=masterdf_ho1_pred2,
                 cOutcome="CAD_tenyearoutcome",
                predrisk1=masterdf_ho1_pred2$framrisk_recal,
                predrisk2=masterdf_ho1_pred2$simpcox51risk,
                cutoff=c(0,0.05,1))

reclass(data=masterdf_ho1_pred2,
                 cOutcome="CAD_tenyearoutcome",
                predrisk1=masterdf_ho1_pred2$tenyearASCVD_recal,
                predrisk2=masterdf_ho1_pred2$simpcox51risk,
                cutoff=c(0,0.05,1))

reclass(data=masterdf_ho1_pred2[!is.na(qrisk3_recal)],
                 cOutcome="CAD_tenyearoutcome",
                predrisk1=masterdf_ho1_pred2[!is.na(qrisk3_recal)]$qrisk3_recal,
                predrisk2=masterdf_ho1_pred2[!is.na(qrisk3_recal)]$simpcox51risk,
                cutoff=c(0,0.05,1))

reclass(data=masterdf_ho1_pred2,
                 cOutcome="CAD_tenyearoutcome",
                predrisk1=masterdf_ho1_pred2$framrisk_recal,
                predrisk2=masterdf_ho1_pred2$simpcox20risk,
                cutoff=c(0,0.025,1))

reclass(data=masterdf_ho1_pred2,
                 cOutcome="CAD_tenyearoutcome",
                predrisk1=masterdf_ho1_pred2$tenyearASCVD_recal,
                predrisk2=masterdf_ho1_pred2$simpcox20risk,
                cutoff=c(0,0.025,1))

reclass(data=masterdf_ho1_pred2[!is.na(qrisk3_recal)],
                 cOutcome="CAD_tenyearoutcome",
                predrisk1=masterdf_ho1_pred2[!is.na(qrisk3_recal)]$qrisk3_recal,
                predrisk2=masterdf_ho1_pred2[!is.na(qrisk3_recal)]$simpcox20risk,
                cutoff=c(0,0.025,1))

reclass(data=masterdf_ho1_pred2,
                 cOutcome="CAD_tenyearoutcome",
                predrisk1=masterdf_ho1_pred2$framrisk_recal,
                predrisk2=masterdf_ho1_pred2$simpcox20risk,
                cutoff=c(0,0.05,1))

reclass(data=masterdf_ho1_pred2,
                 cOutcome="CAD_tenyearoutcome",
                predrisk1=masterdf_ho1_pred2$tenyearASCVD_recal,
                predrisk2=masterdf_ho1_pred2$simpcox20risk,
                cutoff=c(0,0.05,1))

reclass(data=masterdf_ho1_pred2[!is.na(qrisk3_recal)],
                 cOutcome="CAD_tenyearoutcome",
                predrisk1=masterdf_ho1_pred2[!is.na(qrisk3_recal)]$qrisk3_recal,
                predrisk2=masterdf_ho1_pred2[!is.na(qrisk3_recal)]$simpcox20risk,
                cutoff=c(0,0.05,1))

reclass(data=masterdf_ho1_pred2,
                 cOutcome="CAD_tenyearoutcome",
                predrisk1=masterdf_ho1_pred2$framrisk_recal,
                predrisk2=masterdf_ho1_pred2$xgcoxrisk,
                cutoff=c(0,0.025,1))

reclass(data=masterdf_ho1_pred2,
                 cOutcome="CAD_tenyearoutcome",
                predrisk1=masterdf_ho1_pred2$tenyearASCVD_recal,
                predrisk2=masterdf_ho1_pred2$xgcoxrisk,
                cutoff=c(0,0.025,1))

reclass(data=masterdf_ho1_pred2[!is.na(qrisk3_recal)],
                 cOutcome="CAD_tenyearoutcome",
                predrisk1=masterdf_ho1_pred2[!is.na(qrisk3_recal)]$qrisk3_recal,
                predrisk2=masterdf_ho1_pred2[!is.na(qrisk3_recal)]$xgcoxrisk,
                cutoff=c(0,0.025,1))

reclass(data=masterdf_ho1_pred2,
                 cOutcome="CAD_tenyearoutcome",
                predrisk1=masterdf_ho1_pred2$framrisk_recal,
                predrisk2=masterdf_ho1_pred2$xgcoxrisk,
                cutoff=c(0,0.05,1))

reclass(data=masterdf_ho1_pred2,
                 cOutcome="CAD_tenyearoutcome",
                predrisk1=masterdf_ho1_pred2$tenyearASCVD_recal,
                predrisk2=masterdf_ho1_pred2$xgcoxrisk,
                cutoff=c(0,0.05,1))

reclass(data=masterdf_ho1_pred2[!is.na(qrisk3_recal)],
                 cOutcome="CAD_tenyearoutcome",
                predrisk1=masterdf_ho1_pred2[!is.na(qrisk3_recal)]$qrisk3_recal,
                predrisk2=masterdf_ho1_pred2[!is.na(qrisk3_recal)]$xgcoxrisk,
                cutoff=c(0,0.05,1))

dtfordensityplots <- masterdf_ho1_pred2[,c('eid','coxnet51risk','simpcox51risk','xgcoxrisk',
                                   'framrisk','tenyearASCVD','qrisk3','framrisk_recal',
                                   'tenyearASCVD_recal','qrisk3_recal')]

names(dtfordensityplots) <- c('eid','ML4HEN-Cox','SimpleCox51','XgCox','FRS','PCE','QRISK3',
                      'FRS(recal)','PCE(recal)','QRISK(recal)')

densityplotdt1 <- melt(dtfordensityplots,id.vars=c('eid'),measure.vars=c('ML4HEN-Cox'))

densityplotdt2 <- melt(dtfordensityplots,id.vars=c('eid'),measure.vars=c('ML4HEN-Cox',
                                                       'FRS(recal)','PCE(recal)',
                                                       'QRISK(recal)'))

densityplotdt3 <- melt(dtfordensityplots,id.vars=c('eid'),measure.vars=c('ML4HEN-Cox','SimpleCox51',
                                                       'XgCox'))

options(repr.plot.width=12,repr.plot.height=10)

g1 <- ggplot(data=densityplotdt1,aes(x=value,colour=variable)) + 
geom_density() +
scale_x_continuous(limits=c(0,0.15)) +
scale_y_continuous(expand=c(0,0),limits=c(0,45)) +
theme_bw() +
xlab('Predicted Risk') +
labs(colour = 'Model') +
scale_color_manual(values=c('#66C2A5'),labels=c(expression(ML4H[EN-COX]))) +
theme(legend.text.align = 0)

g2 <- ggplot(data=densityplotdt2,aes(x=value,colour=variable)) + 
geom_density() +
theme_bw() +
scale_x_continuous(limits=c(0,0.15)) +
scale_y_continuous(expand=c(0,0),limits=c(0,45)) +
xlab('Predicted Risk') +
labs(colour = 'Model') +
scale_color_manual(values=c('#66C2A5','#FC8D62','#8DA0CB','#E78AC3'),
                   labels=c(expression(ML4H[EN-COX]),'FRS','PCE','QRISK3')) +
theme(legend.text.align = 0)

g3 <- ggplot(data=densityplotdt3,aes(x=value,colour=variable)) + 
geom_density() +
theme_bw() +
scale_x_continuous(limits=c(0,0.15)) +
scale_y_continuous(expand=c(0,0),limits=c(0,45)) +
xlab('Predicted Risk') +
labs(colour = 'Model') +
scale_color_manual(values=c('#66C2A5','#A6D854','#FFD92F'),
                   labels=c(expression(ML4H[EN-COX]),'SimpleCox51','XGBoost')) +
theme(legend.text.align = 0)

absriskplots <- g1 + g2 + g3 + plot_layout(ncol = 1)
absriskplots

# ggsave('/medpop/esp2/sagrawal/mi_mlprediction/2020.11.30_marcus/mlpaperplots/efigure8.pdf',
#        absriskplots,width=12,height=10)

options(repr.plot.width=10,repr.plot.height=10)
densityplotdt4 <- melt(dtfordensityplots,id.vars=c('eid'),measure.vars=c('ML4HEN-Cox','XgCox'))

g4 <- ggplot(data=densityplotdt4,aes(x=value,colour=variable)) + 
geom_density() +
scale_x_continuous(limits=c(0,0.25)) +
theme_classic() +
xlab('Predicted Risk') +
labs(colour = 'Model') +
scale_color_brewer(palette='Set1',labels=c(expression(ML4H[EN-COX]),'XGBoost')) +
theme(text = element_text(size=16),
      legend.position=c(0.8,0.4),
     legend.title=element_blank(),
     legend.text.align=0)

options(repr.plot.width=12,repr.plot.height=12)
efigure11 <- xgCoxFeatPlot / (aucfig2 | g4) + plot_layout(nrow=2,heights = c(2,2))

efigure11

# ggsave('/medpop/esp2/sagrawal/mi_mlprediction/2020.11.30_marcus/mlpaperplots/efigure11.pdf',
#        efigure11,width=10,height=10)

fullfeat_dev <- fread("coxnet__feature_matrix__development_0_0011765691116882482__30Nov_2020.txt")
fullfeat_ho <- fread("coxnet__feature_matrix__holdout_0_0011765691116882482__30Nov_2020.txt")

fullfeat_devUnscaled <- fread('coxnet__feature_matrix_nonzero_model_0_0011765691116882482__development_unscaled__30Nov_2020.txt')
fullfeat_hoUnscaled <- fread('coxnet__feature_matrix_nonzero_model_0_0011765691116882482__holdout_unscaled__30Nov_2020.txt')

prsvec <- quantile(fullfeat_devUnscaled[['d_prsadjscore']],seq(0,1,0.01),na.rm = TRUE)
prsperc <- seq(0,100,1)
        
prsmean <- mean(fullfeat_devUnscaled[['d_prsadjscore']],na.rm=TRUE)
prssd <- sd(fullfeat_devUnscaled[['d_prsadjscore']],na.rm=TRUE)
        
prsvec1 <- (prsvec - prsmean)/prssd
        
agevec <- c(45,55,65)
agevarmean <- mean(fullfeat_devUnscaled[['d_age']],na.rm=TRUE)
agevarsd <- sd(fullfeat_devUnscaled[['d_age']],na.rm=TRUE)
        
prsvec2 <- rep(prsvec1,length(agevec))
agevecrep <- rep(agevec,each=length(prsvec1))
agevecrep_scaled <- (agevecrep - agevarmean)/agevarsd
        
plotdt <- data.table(prsvec2,
                        agevecrep_scaled,
                        rep(prsperc,length(agevec)),
                        agevecrep)
names(plotdt) <- c('d_prsadjscore','d_age','unscaledvar','age_unscaled')
mutnamvec <- names(plotdt)[1:2]
           
modscaleddt <- fullfeat_dev[,-c('index')]
modscaleddtmean <- modscaleddt[,lapply(.SD,mean)]
modscaleddtmeanrep <- modscaleddtmean[rep(seq_len(nrow(modscaleddtmean)),nrow(plotdt))]
        
modscaleddtmeanrep[, (mutnamvec) := plotdt[, .SD, .SDcols=mutnamvec], 
                            .SDcols = mutnamvec]
    
if(!all.equal(coxnet_feature_list_annot_clean$field,names(modscaleddtmeanrep))){
    stop('fields in coefdt and column names in scaleddt not in same order')
}
    
modscaleddtmeanrep[, lp := as.matrix(modscaleddtmeanrep)%*%coxnet_feature_list_annot_clean$coef]
modscaleddtmeanrep[, absrisk := (1 - 0.9876555687885301^exp(lp))]
finaldt <- cbind(modscaleddtmeanrep,
                            plotdt[,.SD,.SDcols=which(names(plotdt) %in% c('unscaledvar','age_unscaled'))])

prsplot <- ggplot(finaldt,aes(x=unscaledvar,y=absrisk*100,
                              color=factor(age_unscaled,levels=c(65,55,45)))) +
    geom_point() +
    scale_y_continuous(limits=c(0,6)) +
    theme_classic() +
    xlab(expression(Adjusted~GPS[CAD]~Percentile)) +
    ylab('Predicted 10-year risk of CAD, %') +
    labs(color = 'Age') +
    scale_color_brewer(palette = 'Set1') +
    theme(text = element_text(size=16),legend.position='none')

hipquant <- quantile(fullfeat_devUnscaled[['d_hip']],c(0.01,0.99),na.rm = TRUE)
hipvec <- seq(hipquant[[1]],hipquant[[2]],length.out=100)
        
hipmean <- mean(fullfeat_devUnscaled[['d_hip']],na.rm=TRUE)
hipsd <- sd(fullfeat_devUnscaled[['d_hip']],na.rm=TRUE)
        
hipvec1 <- (hipvec - hipmean)/hipsd
        
agevec <- c(45,55,65)
agevarmean <- mean(fullfeat_devUnscaled[['d_age']],na.rm=TRUE)
agevarsd <- sd(fullfeat_devUnscaled[['d_age']],na.rm=TRUE)
        
hipvec2 <- rep(hipvec1,length(agevec))
agevecrep <- rep(agevec,each=length(hipvec1))
agevecrep_scaled <- (agevecrep - agevarmean)/agevarsd
        
plotdt <- data.table(hipvec2,
                        agevecrep_scaled,
                        rep(hipvec,length(agevec)),
                        agevecrep)
names(plotdt) <- c('d_hip','d_age','unscaledvar','age_unscaled')
mutnamvec <- names(plotdt)[1:2]
           
modscaleddt <- fullfeat_dev[,-c('index')]
modscaleddtmean <- modscaleddt[,lapply(.SD,mean)]
modscaleddtmeanrep <- modscaleddtmean[rep(seq_len(nrow(modscaleddtmean)),nrow(plotdt))]
        
modscaleddtmeanrep[, (mutnamvec) := plotdt[, .SD, .SDcols=mutnamvec], 
                            .SDcols = mutnamvec]
    
if(!all.equal(coxnet_feature_list_annot_clean$field,names(modscaleddtmeanrep))){
    stop('fields in coefdt and column names in scaleddt not in same order')
}
    
modscaleddtmeanrep[, lp := as.matrix(modscaleddtmeanrep)%*%coxnet_feature_list_annot_clean$coef]
modscaleddtmeanrep[, absrisk := (1 - 0.9876555687885301^exp(lp))]
finaldt <- cbind(modscaleddtmeanrep,
                            plotdt[,.SD,.SDcols=which(names(plotdt) %in% c('unscaledvar','age_unscaled'))])

hipplot <- ggplot(finaldt,aes(x=unscaledvar,y=absrisk*100,
                              color=factor(age_unscaled,levels=c(65,55,45)))) +
    geom_point() +
    scale_y_continuous(limits=c(0,6)) +
    theme_classic() +
    xlab('Hip circumference (cm)') +
    ylab('') +
    labs(color = 'Age') +
    scale_color_brewer(palette = 'Set1') +
    theme(text = element_text(size=16),legend.position='none')

waistquant <- quantile(fullfeat_devUnscaled[['d_waist']],c(0.01,0.99),na.rm = TRUE)
waistvec <- seq(waistquant[[1]],waistquant[[2]],length.out=100)
        
waistmean <- mean(fullfeat_devUnscaled[['d_waist']],na.rm=TRUE)
waistsd <- sd(fullfeat_devUnscaled[['d_waist']],na.rm=TRUE)
        
waistvec1 <- (waistvec - waistmean)/waistsd
        
agevec <- c(45,55,65)
agevarmean <- mean(fullfeat_devUnscaled[['d_age']],na.rm=TRUE)
agevarsd <- sd(fullfeat_devUnscaled[['d_age']],na.rm=TRUE)
        
waistvec2 <- rep(waistvec1,length(agevec))
agevecrep <- rep(agevec,each=length(waistvec1))
agevecrep_scaled <- (agevecrep - agevarmean)/agevarsd
        
plotdt <- data.table(waistvec2,
                        agevecrep_scaled,
                        rep(waistvec,length(agevec)),
                        agevecrep)
names(plotdt) <- c('d_waist','d_age','unscaledvar','age_unscaled')
mutnamvec <- names(plotdt)[1:2]
           
modscaleddt <- fullfeat_dev[,-c('index')]
modscaleddtmean <- modscaleddt[,lapply(.SD,mean)]
modscaleddtmeanrep <- modscaleddtmean[rep(seq_len(nrow(modscaleddtmean)),nrow(plotdt))]
        
modscaleddtmeanrep[, (mutnamvec) := plotdt[, .SD, .SDcols=mutnamvec], 
                            .SDcols = mutnamvec]
    
if(!all.equal(coxnet_feature_list_annot_clean$field,names(modscaleddtmeanrep))){
    stop('fields in coefdt and column names in scaleddt not in same order')
}
    
modscaleddtmeanrep[, lp := as.matrix(modscaleddtmeanrep)%*%coxnet_feature_list_annot_clean$coef]
modscaleddtmeanrep[, absrisk := (1 - 0.9876555687885301^exp(lp))]
finaldt <- cbind(modscaleddtmeanrep,
                            plotdt[,.SD,.SDcols=which(names(plotdt) %in% c('unscaledvar','age_unscaled'))])

waistplot <- ggplot(finaldt,aes(x=unscaledvar,y=absrisk*100,
                              color=factor(age_unscaled,levels=c(65,55,45)))) +
    geom_point() +
    scale_y_continuous(limits=c(0,6)) +
    theme_classic() +
    xlab('Waist circumference (cm)') +
    ylab('') +
    labs(color = 'Age') +
    scale_color_brewer(palette = 'Set1') +
    theme(text = element_text(size=16),legend.position=c(0.7,0.8))

options(repr.plot.width=12,repr.plot.height=12)
figure3 <- featplot / (prsplot | hipplot | waistplot) + plot_layout(nrow=2)
figure3

# ggsave('/medpop/esp2/sagrawal/mi_mlprediction/2020.11.30_marcus/mlpaperplots/figure3.pdf',
#         figure3,width=12,height=12)

# names(plotdt) <- c('d_waist','d_age','unscaledvar','age_unscaled')
# mutnamvec <- names(plotdt)[1:2]
           
modscaleddt <- copy(fullfeat_ho)
modscaleddt[,d_prsadjscore:=0]
    
modscaleddt[, lp := as.matrix(modscaleddt[,-c('index')])%*%coxnet_feature_list_annot_clean$coef]
modscaleddt[, coxnetNoPrsRisk := (1 - 0.9876555687885301^exp(lp))]

masterdf_ho1_pred2 <- merge(masterdf_ho1_pred2,modscaleddt[,c('index','coxnetNoPrsRisk')],by.x='eid',by.y='index')

dim(masterdf_ho1_pred2)
head(masterdf_ho1_pred2)

rocobj_coxnetNoPrs <- roc(masterdf_ho1_pred2$CAD_tenyearoutcome,masterdf_ho1_pred2$coxnetNoPrsRisk)
rocobj_coxnetNoPrs

ci(rocobj_coxnetNoPrs,method='bootstrap',boot.n=1000,progress='text')

reclass(data=masterdf_ho1_pred2,
                 cOutcome="CAD_tenyearoutcome",
                predrisk1=masterdf_ho1_pred2$framrisk_recal,
                predrisk2=masterdf_ho1_pred2$coxnetNoPrsRisk,
                cutoff=c(0,0.025,1))

reclass(data=masterdf_ho1_pred2,
                 cOutcome="CAD_tenyearoutcome",
                predrisk1=masterdf_ho1_pred2$tenyearASCVD_recal,
                predrisk2=masterdf_ho1_pred2$coxnetNoPrsRisk,
                cutoff=c(0,0.025,1))

reclass(data=masterdf_ho1_pred2[!is.na(qrisk3_recal)],
                 cOutcome="CAD_tenyearoutcome",
                predrisk1=masterdf_ho1_pred2[!is.na(qrisk3_recal)]$qrisk3_recal,
                predrisk2=masterdf_ho1_pred2[!is.na(qrisk3_recal)]$coxnetNoPrsRisk,
                cutoff=c(0,0.025,1))

reclass(data=masterdf_ho1_pred2,
                 cOutcome="CAD_tenyearoutcome",
                predrisk1=masterdf_ho1_pred2$framrisk_recal,
                predrisk2=masterdf_ho1_pred2$coxnetNoPrsRisk,
                cutoff=c(0,0.05,1))

reclass(data=masterdf_ho1_pred2,
                 cOutcome="CAD_tenyearoutcome",
                predrisk1=masterdf_ho1_pred2$tenyearASCVD_recal,
                predrisk2=masterdf_ho1_pred2$coxnetNoPrsRisk,
                cutoff=c(0,0.05,1))

reclass(data=masterdf_ho1_pred2[!is.na(qrisk3_recal)],
                 cOutcome="CAD_tenyearoutcome",
                predrisk1=masterdf_ho1_pred2[!is.na(qrisk3_recal)]$qrisk3_recal,
                predrisk2=masterdf_ho1_pred2[!is.na(qrisk3_recal)]$coxnetNoPrsRisk,
                cutoff=c(0,0.05,1))

names(masterdf_ho1_pred2)

masterdf_ho1_pred2[,coxnet51riskQuintiles := quantcut(coxnet51risk,5,labels=paste0('Q',1:5),ordered=TRUE)]
masterdf_ho1_pred2[,coxnet51riskDeciles := quantcut(coxnet51risk,4,labels=paste0('Q',1:4),ordered=TRUE)]

masterdf_ho1_pred2[,simpcox51riskQuintiles := quantcut(simpcox51risk,5,labels=paste0('Q',1:5),ordered=TRUE)]
masterdf_ho1_pred2[,simpcox51riskDeciles := quantcut(simpcox51risk,4,labels=paste0('Q',1:4),ordered=TRUE)]

masterdf_ho1_pred2[,framrisk_recalQuintiles := quantcut(framrisk_recal,5,labels=paste0('Q',1:5),ordered=TRUE)]
masterdf_ho1_pred2[,framrisk_recalDeciles := quantcut(framrisk_recal,4,labels=paste0('Q',1:4),ordered=TRUE)]

masterdf_ho1_pred2[,tenyearASCVD_recalQuintiles := quantcut(tenyearASCVD_recal,5,labels=paste0('Q',1:5),ordered=TRUE)]
masterdf_ho1_pred2[,tenyearASCVD_recalDeciles := quantcut(tenyearASCVD_recal,4,labels=paste0('Q',1:4),ordered=TRUE)]

masterdf_ho1_pred2[,qrisk3_recalQuintiles := quantcut(qrisk3_recal,5,labels=paste0('Q',1:5),ordered=TRUE)]
masterdf_ho1_pred2[,qrisk3_recalDeciles := quantcut(qrisk3_recal,4,labels=paste0('Q',1:4),ordered=TRUE)]

masterdf_ho1_pred2[,framriskQuintiles := quantcut(framrisk,5,labels=paste0('Q',1:5),ordered=TRUE)]
masterdf_ho1_pred2[,framriskDeciles := quantcut(framrisk,4,labels=paste0('Q',1:4),ordered=TRUE)]

masterdf_ho1_pred2[,tenyearASCVDQuintiles := quantcut(tenyearASCVD,5,labels=paste0('Q',1:5),ordered=TRUE)]
masterdf_ho1_pred2[,tenyearASCVDDeciles := quantcut(tenyearASCVD,4,labels=paste0('Q',1:4),ordered=TRUE)]

masterdf_ho1_pred2[,qrisk3Quintiles := quantcut(qrisk3,5,labels=paste0('Q',1:5),ordered=TRUE)]
masterdf_ho1_pred2[,qrisk3Deciles := quantcut(qrisk3,4,labels=paste0('Q',1:4),ordered=TRUE)]

coxnet51riskQuintilesCalc <- 
masterdf_ho1_pred2[order(coxnet51riskQuintiles),
                   lapply(.SD,mean),.SDcols='CAD_tenyearoutcome',
                   by='coxnet51riskQuintiles']
names(coxnet51riskQuintilesCalc) <- c('quant','coxnet51Q20')

coxnet51riskDecilesCalc <- 
masterdf_ho1_pred2[order(coxnet51riskDeciles),
                   lapply(.SD,mean),.SDcols='CAD_tenyearoutcome',
                   by='coxnet51riskDeciles']
names(coxnet51riskDecilesCalc) <- c('quant','coxnet51Q10')



simpcox51riskQuintilesCalc <- 
masterdf_ho1_pred2[order(simpcox51riskQuintiles),
                   lapply(.SD,mean),.SDcols='CAD_tenyearoutcome',
                   by='simpcox51riskQuintiles']
names(simpcox51riskQuintilesCalc) <- c('quant','simpcox51Q20')

simpcox51riskDecilesCalc <- 
masterdf_ho1_pred2[order(simpcox51riskDeciles),
                   lapply(.SD,mean),.SDcols='CAD_tenyearoutcome',
                   by='simpcox51riskDeciles']
names(simpcox51riskDecilesCalc) <- c('quant','simpcox51Q10')




framrisk_recalQuintilesCalc <- 
masterdf_ho1_pred2[order(framrisk_recalQuintiles),
                   lapply(.SD,mean),.SDcols='CAD_tenyearoutcome',
                   by='framrisk_recalQuintiles']
names(framrisk_recalQuintilesCalc) <- c('quant','framriskQ20')

framrisk_recalDecilesCalc <- 
masterdf_ho1_pred2[order(framrisk_recalDeciles),
                   lapply(.SD,mean),.SDcols='CAD_tenyearoutcome',
                   by='framrisk_recalDeciles']
names(framrisk_recalDecilesCalc) <- c('quant','framriskQ10')



tenyearASCVD_recalQuintilesCalc <- 
masterdf_ho1_pred2[order(tenyearASCVD_recalQuintiles),
                   lapply(.SD,mean),.SDcols='CAD_tenyearoutcome',
                   by='tenyearASCVD_recalQuintiles']
names(tenyearASCVD_recalQuintilesCalc) <- c('quant','PCEQ20')

tenyearASCVD_recalDecilesCalc <- 
masterdf_ho1_pred2[order(tenyearASCVD_recalDeciles),
                   lapply(.SD,mean),.SDcols='CAD_tenyearoutcome',
                   by='tenyearASCVD_recalDeciles']
names(tenyearASCVD_recalDecilesCalc) <- c('quant','PCEQ10')



qrisk3_recalQuintilesCalc <- 
masterdf_ho1_pred2[order(qrisk3_recalQuintiles),
                   lapply(.SD,mean),.SDcols='CAD_tenyearoutcome',
                   by='qrisk3_recalQuintiles']
names(qrisk3_recalQuintilesCalc) <- c('quant','qrisk3Q20')
qrisk3_recalQuintilesCalc <- qrisk3_recalQuintilesCalc[!is.na(quant)]

qrisk3_recalDecilesCalc <- 
masterdf_ho1_pred2[order(qrisk3_recalDeciles),
                   lapply(.SD,mean),.SDcols='CAD_tenyearoutcome',
                   by='qrisk3_recalDeciles']
names(qrisk3_recalDecilesCalc) <- c('quant','qrisk3Q10')
qrisk3_recalDecilesCalc <- qrisk3_recalDecilesCalc[!is.na(quant)]


framriskQuintilesCalc <- 
masterdf_ho1_pred2[order(framriskQuintiles),
                   lapply(.SD,mean),.SDcols='CAD_tenyearoutcome',
                   by='framriskQuintiles']
names(framriskQuintilesCalc) <- c('quant','framriskRawQ20')

framriskDecilesCalc <- 
masterdf_ho1_pred2[order(framriskDeciles),
                   lapply(.SD,mean),.SDcols='CAD_tenyearoutcome',
                   by='framriskDeciles']
names(framriskDecilesCalc) <- c('quant','framriskRawQ10')



tenyearASCVDQuintilesCalc <- 
masterdf_ho1_pred2[order(tenyearASCVDQuintiles),
                   lapply(.SD,mean),.SDcols='CAD_tenyearoutcome',
                   by='tenyearASCVDQuintiles']
names(tenyearASCVDQuintilesCalc) <- c('quant','PCERawQ20')

tenyearASCVDDecilesCalc <- 
masterdf_ho1_pred2[order(tenyearASCVDDeciles),
                   lapply(.SD,mean),.SDcols='CAD_tenyearoutcome',
                   by='tenyearASCVDDeciles']
names(tenyearASCVDDecilesCalc) <- c('quant','PCERawQ10')



qrisk3QuintilesCalc <- 
masterdf_ho1_pred2[order(qrisk3Quintiles),
                   lapply(.SD,mean),.SDcols='CAD_tenyearoutcome',
                   by='qrisk3Quintiles']
names(qrisk3QuintilesCalc) <- c('quant','qrisk3RawQ20')
qrisk3QuintilesCalc <- qrisk3QuintilesCalc[!is.na(quant)]

qrisk3DecilesCalc <- 
masterdf_ho1_pred2[order(qrisk3Deciles),
                   lapply(.SD,mean),.SDcols='CAD_tenyearoutcome',
                   by='qrisk3Deciles']
names(qrisk3DecilesCalc) <- c('quant','qrisk3RawQ10')
qrisk3DecilesCalc <- qrisk3DecilesCalc[!is.na(quant)]

Quintilesdt <- merge(coxnet51riskQuintilesCalc,simpcox51riskQuintilesCalc,by='quant')
Quintilesdt <- merge(Quintilesdt,framrisk_recalQuintilesCalc,by='quant')
Quintilesdt <- merge(Quintilesdt,tenyearASCVD_recalQuintilesCalc,by='quant')
Quintilesdt <- merge(Quintilesdt,qrisk3_recalQuintilesCalc,by='quant')
Quintilesdt <- merge(Quintilesdt,framriskQuintilesCalc,by='quant')
Quintilesdt <- merge(Quintilesdt,tenyearASCVDQuintilesCalc,by='quant')
Quintilesdt <- merge(Quintilesdt,qrisk3QuintilesCalc,by='quant')

Quintilesdtmelted <- melt(Quintilesdt,id.vars=c('quant'),
                        measure.vars=c('coxnet51Q20','simpcox51Q20',
                                      'framriskQ20','PCEQ20','qrisk3Q20',
                                      'framriskRawQ20','PCERawQ20','qrisk3RawQ20'))

Quintilesdtmelted$variable <- factor(Quintilesdtmelted$variable,
                                  levels=c('coxnet51Q20','simpcox51Q20',
                                      'framriskQ20','PCEQ20','qrisk3Q20',
                                          'framriskRawQ20','PCERawQ20','qrisk3RawQ20'),
                                  ordered=TRUE,labels=c('ML4HEN-Cox','SimpleCox51',
                                                       'FRS(recal)','PCE(recal)',
                                                       'QRISK(recal)','FRS(raw)',
                                                       'PCE(raw)','QRISK(raw)'))

QuintilesdtmeltedFig4 <- melt(Quintilesdt,id.vars=c('quant'),
                        measure.vars=c('coxnet51Q20',
                                      'framriskQ20','PCEQ20','qrisk3Q20'))

QuintilesdtmeltedFig4$variable <- factor(QuintilesdtmeltedFig4$variable,
                                  levels=c('coxnet51Q20',
                                      'framriskQ20','PCEQ20','qrisk3Q20'),
                                  ordered=TRUE,labels=c('ML4HEN-Cox',
                                                       'FRS','PCE',
                                                       'QRISK3'))

options(repr.plot.width=8,repr.plot.height=8)
QuintilesgradFig4 <- ggplot(QuintilesdtmeltedFig4,aes(x=quant,y=value*100,
                              fill=variable)) +
    geom_bar(stat='identity',width=0.8,position='dodge') +
    scale_y_continuous(limits=c(0,10.5),expand=c(0,0)) +
    theme_classic() +
    xlab('Quintiles of Predicted Risk') +
    ylab('Observed 10-year risk of CAD, %') +
    labs(fill = 'Model') +
    scale_fill_brewer(palette = 'Set2',labels=c(expression(ML4H[EN-COX]),'FRS','PCE','QRISK3')) +
    theme(axis.title.y = element_text(size=18),
          axis.text.y = element_text(size=16,color='black'),
          axis.ticks.y = element_blank(),
          axis.title.x = element_text(size=18),
          axis.text.x = element_text(size=16,color='black'),
          axis.ticks.x = element_blank(),
          legend.position=c(0.2,0.8),
          legend.text.align=0,
         legend.key.size=unit(1,'cm'),
         legend.title = element_text(size=16),
         legend.text = element_text(size=16))

QuintilesgradFig4

# ggsave('/medpop/esp2/sagrawal/mi_mlprediction/2020.11.30_marcus/mlpaperplots/figure4_gradient.pdf',
#         QuintilesgradFig4,width=8,height=8)
