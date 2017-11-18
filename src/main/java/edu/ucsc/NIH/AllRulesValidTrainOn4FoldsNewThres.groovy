/*
 * This file is part of the PSL software.
 * Copyright 2011-2013 University of Maryland
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package edu.ucsc.NIH;

import java.util.Comparator;

import edu.umd.cs.psl.application.learning.weight.maxlikelihood.LazyMaxLikelihoodMPE;
import edu.umd.cs.psl.config.*
import edu.umd.cs.psl.ui.functions.textsimilarity.*
import edu.umd.cs.psl.application.inference.MPEInference;
import edu.umd.cs.psl.application.learning.weight.maxlikelihood.MaxLikelihoodMPE;
import edu.umd.cs.psl.database.DataStore
import edu.umd.cs.psl.database.Database;
import edu.umd.cs.psl.database.DatabasePopulator;
import edu.umd.cs.psl.database.Partition;
import edu.umd.cs.psl.database.ReadOnlyDatabase;
import edu.umd.cs.psl.database.rdbms.RDBMSDataStore
import edu.umd.cs.psl.database.rdbms.RDBMSUniqueStringID
import edu.umd.cs.psl.database.rdbms.driver.H2DatabaseDriver
import edu.umd.cs.psl.database.rdbms.driver.H2DatabaseDriver.Type
import edu.umd.cs.psl.evaluation.result.memory.MemoryFullInferenceResult
import edu.umd.cs.psl.groovy.PSLModel;
import edu.umd.cs.psl.groovy.PredicateConstraint;
import edu.umd.cs.psl.groovy.SetComparison;
import edu.umd.cs.psl.model.argument.ArgumentType;
import edu.umd.cs.psl.model.argument.GroundTerm;
import edu.umd.cs.psl.model.argument.UniqueID;
import edu.umd.cs.psl.model.argument.Variable;
import edu.umd.cs.psl.model.atom.GroundAtom;
import edu.umd.cs.psl.model.function.ExternalFunction;
import edu.umd.cs.psl.ui.loading.InserterUtils;
import edu.umd.cs.psl.util.database.Queries;
import edu.ucsc.NIH.EvalResults;
import edu.ucsc.NIH.EvalResults.EvalResultsSingle;
import edu.ucsc.NIH.EvalResults.Matches;
import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.Multimap;
import com.google.common.collect.Ordering;
import com.google.common.collect.TreeMultimap;
import java.util.Comparator;
import java.util.Map;
import java.util.Map.Entry;



ConfigManager cm = ConfigManager.getManager()
ConfigBundle config = cm.getBundle("nih-transitivity")

TimeNow = new Date();

def defaultPath = System.getProperty("java.io.tmpdir")
String dbpath = "./nih";
//String dir = 'data' + java.io.File.separator+'NIH' + java.io.File.separator + '5folds' + java.io.File.separator + '5foldsValidation' + java.io.File.separator;
String dir = 'data' + java.io.File.separator+'NIH' + java.io.File.separator;
String namesDir = dir;
String relationalSimDir = dir;
String transRelationalDir = dir;
String resultsFileName = "final.NTB";
String resultsFile = dir + "resultsFiles" + java.io.File.separator + resultsFileName ;

//squared or linear potentials
sqPotentials = true;
oneDirPotentials = false;

//these are to compute the final statistics from 5 folds:
//precision, recall, f-measure and Standard Deviation for all metrics
edu.ucsc.NIH.EvalResults  results = new EvalResults();
edu.ucsc.NIH.EvalResults  results1_1 = new EvalResults();


//this is to store the results after the application of the 1-1 matching
edu.ucsc.NIH.EvalResults  resultsOneToOne = new EvalResults();


int numFolds = 5
double thresholdStep = 0.001

//we learn the initial weights from a grid search
int posFirstNameWeight = 10;
int posLastNameWeight = 10;
int posMaidenNameWeight = 1;

double negFirstNameWeight = 1;
double negLastNameWeight = 1;
double negMaidenNameWeight = 0.001;

int ageWeight = 10;
int negAgeWeight = 1;
int negGenderLivingWeight =10;

double firstDegreeSimWeight = 2;
double secDegreeSimWeight = 1;
double firstDegreeTransWeight = 15;
double negFirstDegreeTransWeight = 1;
double secDegreeTransWeight = 1;

double priorWeight = 12;
double bijectionWeight = 0.1;
double transWeight = 1;

//to run the different versions of the PSL model make sure to change accordingly these parameters from true to false
boolean persInfRun = true;
boolean firstDegSimRun = true;
boolean secDegSimRun = true;
boolean firstDegTranRun = true;
boolean secDegTranRun = true;
boolean bijRun = true;
boolean transRun = true;

println "START TIME = " + new Date()

DataStore data = new RDBMSDataStore(new H2DatabaseDriver(Type.Disk, dbpath, true), config)
//this is the id for the new Partition
int partitionID = 0;
Partition evidencePartitionCommon = new Partition(partitionID++);


for(int fold=1;fold<=numFolds;fold++){
	
		if(fold==2 || fold==3)
		priorWeight=10
	else if(fold==5)
		priorWeight=8
			
	//create the model, define the predicates and the rules
	PSLModel m = createModel(this, data, 
		posFirstNameWeight, posLastNameWeight, posMaidenNameWeight,
		negFirstNameWeight, negLastNameWeight, negMaidenNameWeight,
		ageWeight, negAgeWeight, negGenderLivingWeight,
		firstDegreeSimWeight, secDegreeSimWeight,
	    firstDegreeTransWeight,  secDegreeTransWeight,  negFirstDegreeTransWeight,
		bijectionWeight,  transWeight,  priorWeight,
		persInfRun,  firstDegSimRun,
		secDegSimRun,  firstDegTranRun,  secDegTranRun,
		bijRun, transRun)
	
	println m;
	
	println "Run for fold " + fold;
	
	println "About to load the persons"
	println "TIME before reading the files " + new Date();
	
	//in evidence partition we insert the data with similarities, and general info like belongsToFamily
	//in target partition we insert everything that we need to predict 
	//these 3 folds are for the weight learning
	Partition evidencePartitionWL1 = new Partition(partitionID++);// observed data for weight learning
	Partition targetPartitionWL1 = new Partition(partitionID++);// unobserved data for weight learning
	Partition trueDataPartitionWL1 = new Partition(partitionID++);  // train set for inference
	
	//this fold is for validating (finding the right threshold)
	Partition evidencePartitionValid = new Partition(partitionID++);		 // test set for validation
	Partition targetPartitionValid = new Partition(partitionID++);
	
	//this fold is for testing
	Partition evidencePartitionTest = new Partition(partitionID++);		 // test set for inference
	Partition targetPartitionTest = new Partition(partitionID++);
	
	if(fold==1){
		//load boolean data
		def insert = data.getInserter(person, evidencePartitionCommon)
		InserterUtils.loadDelimitedData(insert, dir+"persons");
		
		println "About to load the belongsToFamily"
		insert = data.getInserter(belongsToFamily, evidencePartitionCommon)
		InserterUtils.loadDelimitedData(insert, dir+"belongsToFamily");
		
		println "About to load the belongsToParticipant"
		insert = data.getInserter(belongsToParticipant, evidencePartitionCommon)
		InserterUtils.loadDelimitedData(insert, dir+"belongsToParticipant");
		
		println "About to load the Jaro similarities"
		//Load Jaro distance similarities
		insert = data.getInserter(firstNameJaroSim, evidencePartitionCommon)
		InserterUtils.loadDelimitedDataTruth(insert, namesDir +"firstNameJaroSims");
		
		insert = data.getInserter(maidenNameJaroSim, evidencePartitionCommon)
		InserterUtils.loadDelimitedDataTruth(insert, namesDir+"maidenNameJaroSims");
		
		insert = data.getInserter(lastNameJaroSim, evidencePartitionCommon)
		InserterUtils.loadDelimitedDataTruth(insert, namesDir+"lastNameJaroSims");
		
		println "About to load the Levenshtein similarities"
		//Load Levenshtein Similarities
		insert = data.getInserter(firstNameLevSim, evidencePartitionCommon)
		InserterUtils.loadDelimitedDataTruth(insert, namesDir+"firstNameLevSims");
		
		insert = data.getInserter(maidenNameLevSim, evidencePartitionCommon)
		InserterUtils.loadDelimitedDataTruth(insert, namesDir+"maidenNameLevSims");
		
		insert = data.getInserter(lastNameLevSim, evidencePartitionCommon)
		InserterUtils.loadDelimitedDataTruth(insert, namesDir+"lastNameLevSims");
		
		//load the personal information files
		println "About to load the genders"
		insert = data.getInserter(sameGender, evidencePartitionCommon)
		InserterUtils.loadDelimitedDataTruth(insert, dir+"genders");
		
		if(persInfRun){
			println "About to load the ages"
			insert = data.getInserter(similarAge, evidencePartitionCommon);
			InserterUtils.loadDelimitedDataTruth(insert, dir+"contRatioAges");
			
			println "About to load the livingstatus"
			insert = data.getInserter(sameLivingStatus, evidencePartitionCommon)
			InserterUtils.loadDelimitedDataTruth(insert, dir+"livingstatus");
		}
		
		if(firstDegSimRun){
			println "About to load the 1st degree Relationship similarities"
			insert = data.getInserter(sistersSim, evidencePartitionCommon)
			InserterUtils.loadDelimitedDataTruth(insert, relationalSimDir+"sistersSim");
			
			insert = data.getInserter(brothersSim, evidencePartitionCommon)
			InserterUtils.loadDelimitedDataTruth(insert, relationalSimDir+"brothersSim");
			
			insert = data.getInserter(fatherSim, evidencePartitionCommon)
			InserterUtils.loadDelimitedDataTruth(insert, relationalSimDir+"fatherSim");
			
			insert = data.getInserter(motherSim, evidencePartitionCommon)
			InserterUtils.loadDelimitedDataTruth(insert, relationalSimDir+"motherSim");
			
			insert = data.getInserter(daughtersSim, evidencePartitionCommon)
			InserterUtils.loadDelimitedDataTruth(insert, relationalSimDir+"daughtersSim");
			
			insert = data.getInserter(sonsSim, evidencePartitionCommon)
			InserterUtils.loadDelimitedDataTruth(insert, relationalSimDir+"sonsSim");
			
			insert = data.getInserter(spouseSim, evidencePartitionCommon)
			InserterUtils.loadDelimitedDataTruth(insert, relationalSimDir+"spouseSim");
		}
		
		if(secDegSimRun){
			//load second degree relationship similarities
			println "About to load the 2nd degree Relationship files"
			insert = data.getInserter(grandFathersSim, evidencePartitionCommon)
			InserterUtils.loadDelimitedDataTruth(insert, relationalSimDir+"grandFathersSim");
			
			insert = data.getInserter(grandMothersSim, evidencePartitionCommon)
			InserterUtils.loadDelimitedDataTruth(insert, relationalSimDir+"grandMothersSim");
			
			insert = data.getInserter(grandDaughtersSim, evidencePartitionCommon)
			InserterUtils.loadDelimitedDataTruth(insert, relationalSimDir+"grandDaughtersSim");
			
			insert = data.getInserter(grandSonsSim, evidencePartitionCommon)
			InserterUtils.loadDelimitedDataTruth(insert, relationalSimDir+"grandSonsSim");
			
			insert = data.getInserter(unclesSim, evidencePartitionCommon)
			InserterUtils.loadDelimitedDataTruth(insert, relationalSimDir+"unclesSim");
			
			insert = data.getInserter(auntsSim, evidencePartitionCommon)
			InserterUtils.loadDelimitedDataTruth(insert, relationalSimDir+"auntsSim");
			
			insert = data.getInserter(niecesSim, evidencePartitionCommon)
			InserterUtils.loadDelimitedDataTruth(insert, relationalSimDir+"niecesSim");
			
			insert = data.getInserter(nephewsSim, evidencePartitionCommon)
			InserterUtils.loadDelimitedDataTruth(insert, relationalSimDir+"nephewsSim");
		}
		
		if(firstDegTranRun){
			println "About to load the 1st degree Relationship files"
			
			insert = data.getInserter(hasSister, evidencePartitionCommon)
			InserterUtils.loadDelimitedData(insert, dir+"hasSister");
			
			insert = data.getInserter(hasBrother, evidencePartitionCommon)
			InserterUtils.loadDelimitedData(insert, dir+"hasBrother");
			
			insert = data.getInserter(hasFather, evidencePartitionCommon)
			InserterUtils.loadDelimitedData(insert, dir+"hasFather");
			
			insert = data.getInserter(hasMother, evidencePartitionCommon)
			InserterUtils.loadDelimitedData(insert, dir+"hasMother");
			
			insert = data.getInserter(hasDaughter, evidencePartitionCommon)
			InserterUtils.loadDelimitedData(insert, dir+"hasDaughter");
			
			insert = data.getInserter(hasSon, evidencePartitionCommon)
			InserterUtils.loadDelimitedData(insert, dir+"hasSon");
			
			insert = data.getInserter(hasSpouse, evidencePartitionCommon)
			InserterUtils.loadDelimitedData(insert, dir+"hasSpouse");
		}
		
		if(secDegTranRun){
			println "About to load the 2nd degree Relationship files"
			insert = data.getInserter(hasGrandFather, evidencePartitionCommon)
			InserterUtils.loadDelimitedData(insert, dir+"hasGrandFather");
			
			insert = data.getInserter(hasGrandMother, evidencePartitionCommon)
			InserterUtils.loadDelimitedData(insert, dir+"hasGrandMother");
			
			insert = data.getInserter(hasGrandDaughter, evidencePartitionCommon)
			InserterUtils.loadDelimitedData(insert, dir+"hasGrandDaughter");
			
			insert = data.getInserter(hasGrandSon, evidencePartitionCommon)
			InserterUtils.loadDelimitedData(insert, dir+"hasGrandSon");
			
			insert = data.getInserter(hasUncle, evidencePartitionCommon)
			InserterUtils.loadDelimitedData(insert, dir+"hasUncle");
			
			insert = data.getInserter(hasAunt, evidencePartitionCommon)
			InserterUtils.loadDelimitedData(insert, dir+"hasAunt");
			
			insert = data.getInserter(hasNiece, evidencePartitionCommon)
			InserterUtils.loadDelimitedData(insert, dir+"hasNiece");
			
			insert = data.getInserter(hasNephew, evidencePartitionCommon)
			InserterUtils.loadDelimitedData(insert, dir+"hasNephew");
		}
		
		if(firstDegTranRun || secDegTranRun){
			insert = data.getInserter(maxJaroLevFirstName, evidencePartitionCommon)
			InserterUtils.loadDelimitedDataTruth(insert, transRelationalDir + "maxJaroLevFirstName");
			
		}
		
	}
	
	println "About to load the blocking predicates and the predicates we want to predict"
	//load data for the weight learning
	insert = data.getInserter(samePersonBlock, evidencePartitionWL1)
	InserterUtils.loadDelimitedData(insert, dir+"samePersonBlockWL1.fold"+fold);
	
	//put everything you want to predict to the targetPartition
	insert = data.getInserter(samePerson, targetPartitionWL1)
	InserterUtils.loadDelimitedData(insert, dir + "samePersonBlockWL1.fold"+fold);
	
	insert = data.getInserter(samePerson, trueDataPartitionWL1)
	InserterUtils.loadDelimitedDataTruth(insert, dir + "samePersonTrainTruthWL1.fold" + fold);
	
	println "TIME after reading the files " + new Date();
	
	//call the function that performs weight learning
	performWeightLearning(m, data, config,
		 evidencePartitionCommon,  targetPartitionWL1,evidencePartitionWL1, trueDataPartitionWL1);
	
	
	//COMPUTE THE THRESHOLD USING THE VALIDATION SET
	println "About to do validation to find the threshold value"
	insert = data.getInserter(samePersonBlock, evidencePartitionValid)
	InserterUtils.loadDelimitedData(insert, dir+"samePersonValidBlock.fold"+fold);
	
	println "About to load the toPredict predicate to the target partition, i.e. we predict all the pairs (direct and inverse)"
	//put everything you want to predict to the targetPartition
	insert = data.getInserter(samePerson, targetPartitionValid)
	InserterUtils.loadDelimitedData(insert, dir + "samePersonValidBlock.fold"+fold);
	
	
	//dbValid is for validation only db
	Database dbValid = data.getDatabase(targetPartitionValid,
		[person, samePersonBlock, belongsToFamily,belongsToParticipant, 
			similarAge, sameGender, sameLivingStatus,
			firstNameJaroSim, maidenNameJaroSim, lastNameJaroSim, 
			firstNameLevSim, maidenNameLevSim, lastNameLevSim,
			maxJaroLevFirstName, //transitiveSimFirstName,
			motherSim, fatherSim, daughtersSim, sonsSim, sistersSim, brothersSim, spouseSim,//first degree
			grandMothersSim, grandFathersSim, grandDaughtersSim, grandSonsSim, auntsSim, unclesSim, niecesSim, nephewsSim, //second degree
			hasMother, hasFather, hasSister, hasBrother, hasSister, hasDaughter, hasSon, hasSpouse,
			hasGrandMother, hasGrandFather, hasGrandDaughter, hasGrandSon, hasUncle, hasAunt, hasNephew, hasNiece
			
		] as Set, evidencePartitionCommon, evidencePartitionValid);
	
	println "TIME before performing MPE inference " + new Date();
	
	//run MPE inference with learned weights
	MPEInference inferenceApp = new MPEInference(m, dbValid, config);
	MemoryFullInferenceResult inf_result = inferenceApp.mpeInference();
	
	inferenceApp.close();
	
	println "TIME after performing MPE inference " + new Date();
	
	String groundTruthName = dir + "samePersonValidTruth.fold" + fold;
	
	Matches matches = new Matches();
	//below each entry of this hashmap is <"person1+,person2",value> (e.g. we concatenate with , the person1 with person2)
	//call the load ground truth
	HashMap<String, Double> persons1_persons2_matchvalues = loadGroundTruth(groundTruthName, matches)
	//call the load pred values
	HashMap<String, Double> persons1_persons2_predvalues = loadPredictedValues(dbValid)
	dbValid.close(); //we will not need the DB again
	
	//////1-1 change
	//here apply the 1-1 algorithm
	//first we need to store the persons1_persons2_predvalues structure in the correct order
	//i.e. store it in a treemultimap structure where the elements are sorted in
	//decreasing probability value
	TreeMultimap<Double, String> sortedProbabilityPersonsMatched = parsePredictions(persons1_persons2_predvalues, 0) //the 0 is the best threshold value - for now we store all the variables that have sim above 0
	
	//here we need to parse the file with the belongsToParticipant information
	String belongsToPartFile = dir + "belongsToParticipant"
	HashMap<String, String> personID_participantID = loadBelongsToParticipant(belongsToPartFile);
	
	//now perform the 1-1 matching algorithm given the structures
	//sortedProbabilityPersonsMatched and persons1_persons2_matchvalues
	HashMap<String, Double> persons1_persons2_predvalues1_1 = GreedyOneToOneMatching(sortedProbabilityPersonsMatched, personID_participantID);

	double bestThreshold = findThreshold(persons1_persons2_matchvalues, persons1_persons2_predvalues1_1,
		matches, thresholdStep)
	
	//now that you learned the threshold, run again weight learning using all 4 folds 
	//in order to learn a better model than the one that used only the 3 folds
	
	//NOW DO THE FINAL RUN TO THE FINAL FOLD with the bestThreshold learned value
	insert = data.getInserter(samePersonBlock, evidencePartitionTest)
	InserterUtils.loadDelimitedData(insert, dir+"toPredict.fold"+fold);
	
	println "About to load the toPredict predicate to the target partition, i.e. we predict all the pairs (direct and inverse)"
	//put everything you want to predict to the targetPartition
	insert = data.getInserter(samePerson, targetPartitionTest)
	InserterUtils.loadDelimitedData(insert, dir + "toPredict.fold"+fold);
	
	TimeNow = new Date();
	println "Time after reading the files = " + TimeNow
	
	println "[DEBUG]: Now starting the inference"
	//dbInf is for inference only
	Database dbInf = data.getDatabase(targetPartitionTest,
		[person, samePersonBlock, belongsToFamily,belongsToParticipant, 
			similarAge, sameGender, sameLivingStatus,
			firstNameJaroSim, maidenNameJaroSim, lastNameJaroSim, 
			firstNameLevSim, maidenNameLevSim, lastNameLevSim,
			maxJaroLevFirstName,
			motherSim, fatherSim, daughtersSim, sonsSim, sistersSim, brothersSim, spouseSim,//first degree
			grandMothersSim, grandFathersSim, grandDaughtersSim, grandSonsSim, auntsSim, unclesSim, niecesSim, nephewsSim, //second degree
			hasMother, hasFather, hasSister, hasBrother, hasSister, hasDaughter, hasSon, hasSpouse,
			hasGrandMother, hasGrandFather, hasGrandDaughter, hasGrandSon, hasUncle, hasAunt, hasNephew, hasNiece
		] as Set, evidencePartitionCommon, evidencePartitionTest);
	
	println "TIME before performing MPE inference for the second time " + new Date();
	
	//run MPE inference with learned weights
	inferenceApp = new MPEInference(m, dbInf, config);
	inf_result = inferenceApp.mpeInference();
	
	inferenceApp.close();

	println "TIME after performing MPE inference for the second time " + new Date();
	
	groundTruthName = dir + "samePersonTestTruth.fold" + fold;
	matches = new Matches();
	//below each entry of this hashmap is <"person1,person2",value> (e.g. we concatenate with , the person1 with person2)
	//call the load ground truth
	persons1_persons2_matchvalues = loadGroundTruth(groundTruthName, matches)
	//call the load pred values
	persons1_persons2_predvalues = loadPredictedValues(dbInf)
	dbInf.close(); //we will not need the DB again
	
	//we need to run the function computeConfusionMatrix with the best threshold value
	EvalResultsSingle bestResults = new EvalResultsSingle();
	ConfusionMatrix confMat = computeConfusionMatrix(persons1_persons2_predvalues, persons1_persons2_matchvalues, bestThreshold, resultsFile + ".fold" + fold);
	computeStatsOneFold(confMat, matches, bestResults);
	println "Inference -- Best Results:\nPrecision = " + bestResults.precision
	println "Recall = " + bestResults.recall
	println "FMeasure = " + bestResults.fmeasure

	//this is to update the evalResults class
	updateStatsAllFolds(results, confMat, matches);
	TimeNow = new Date();
	
	
	//////1-1 change
	//here apply the 1-1 algorithm
	//first we need to store the persons1_persons2_predvalues structure in the correct order
	//i.e. store it in a treemultimap structure where the elements are sorted in
	//decreasing probability value
	sortedProbabilityPersonsMatched = parsePredictions(persons1_persons2_predvalues, bestThreshold) //the 0 is the best threshold value - for now we store all the variables that have sim above 0
	
	//now perform the 1-1 matching algorithm given the structures
	//sortedProbabilityPersonsMatched and persons1_persons2_matchvalues
	persons1_persons2_predvalues1_1 = GreedyOneToOneMatching(sortedProbabilityPersonsMatched, personID_participantID);

	
	println "About to update the confusion matrix 1-1"
	ConfusionMatrix confMat1_1 = computeConfusionMatrix1_1(persons1_persons2_predvalues1_1, persons1_persons2_matchvalues)
	EvalResultsSingle bestResults1_1 = new EvalResultsSingle();
	computeStatsOneFold(confMat1_1, matches, bestResults1_1);
	println "1-1 -- Results:\nPrecision = " + bestResults1_1.precision
	println "Recall = " + bestResults1_1.recall
	println "FMeasure = " + bestResults1_1.fmeasure

	//this is to update the evalResults class
	updateStatsAllFolds(results1_1, confMat1_1, matches);
	
	println "Time at the end of fold " + fold + " = " + TimeNow
	
	//call the garbage collector - just in case!
	System.gc();
		
}

//now variable results should have the final values
//and then print the results
results.printResults3Digits()

println "After 1-1 Matching"
results1_1.printResults3Digits()


TimeNow = new Date();
println "END TIME = " + TimeNow

//this function takes as input the belongsToParticipant file
//and puts all this information into the structure personID_participantID
HashMap<String, String> loadBelongsToParticipant(String fileName){
	
	HashMap<String, String> personID_participantID = new HashMap()
	
	def labels = new File(fileName)
	def words, personID, participantID
	labels.eachLine {
		line ->
		words = line.split("\t")
		personID=words[0].toString();
		participantID=words[1].toString();
		
		personID_participantID.put(personID, participantID);
		
	}
	return personID_participantID;
}


//this function takes as input the sorted (in terms of sim value) list sortedProbabilityPersonsMatched
//and returns as output the structure persons1_persons2_predvalues1_1 
//which has only the pairs that satisfy the 1-1 restriction
HashMap<String, Double> GreedyOneToOneMatching(
	TreeMultimap<Double, String> sortedProbabilityPersonsMatched, 
	HashMap<String, String> personID_participantID){
	
	
	System.out.println("[DEBUG]: Start GreedyOneToOneMatching");
	
	//this is the structure that we will return that stores only one copy of each decision
	HashMap<String, Double> persons1_persons2_predvalues1_1 = new HashMap()
	
	//this structure stores for each person the other persons that he has already been resolved with
	//an example entry is : <123,456>
	//note that the reverse entry  will be stored as well eg <456,123>
	Multimap<String, String> PersonsMatched = new ArrayListMultimap();
	
	int countOfMatches=0;
	//first iterate over the sortedSimilarityPersonsMatched structure
	Map<Double, Collection<String>> map = sortedProbabilityPersonsMatched.asMap();
	for(Map.Entry<Double, Collection<String>> entry : map.entrySet()){
		Double probability = entry.getKey();
		//for all the pair of persons who happen to have the same similarity value
		for(String matchedPersons : entry.getValue()){
			String[] persons = matchedPersons.split(",");
			String person1 = persons[0];
			String person2 = persons[1];
			String barcode1 = personID_participantID.get(person1)
			String barcode2 = personID_participantID.get(person2)
			
			//search for both of the persons in the structure PersonsMatched
			//if none of the two names are in there this means that this pair is a true match
			//so print it and put it in the PersonsMatched structure
			if(!PersonsMatched.containsKey(person1) && !PersonsMatched.containsKey(person2)){
					//put the pair of persons into the PersonsMatched structure iff similarity>threshold
					countOfMatches++;
					PersonsMatched.put(person1, person2);
					PersonsMatched.put(person2, person1);
					//insert the entry to the structure we are about to return 
					persons1_persons2_predvalues1_1.put(person1+","+person2, probability)
					persons1_persons2_predvalues1_1.put(person2+","+person1, probability)
			}
			else{
				boolean breakSignal=false;
				Collection<String> matchedPersons1 = PersonsMatched.get(person1);
				for(String matchedPerson : matchedPersons1){
					String barcodeMatchedPerson = personID_participantID.get(matchedPerson)
					if(barcodeMatchedPerson.equals(barcode2)){
						breakSignal=true;
						break;
					}
				}
				if(breakSignal) continue;
				Collection<String> matchedPersons2 = PersonsMatched.get(person2);
				for(String matchedPerson : matchedPersons2){
					String barcodeMatchedPerson = personID_participantID.get(matchedPerson)
					if(barcodeMatchedPerson.equals(barcode1)){
						breakSignal=true;
						break;
					}
				}
				if(breakSignal) continue;
				//if we reach this point it means that we have not found a match yet for this pair of persons
				//so we need to include it in the structure
				countOfMatches++;
				PersonsMatched.put(person1, person2);
				PersonsMatched.put(person2, person1);
				//insert this entry to the structure that we will return 
				persons1_persons2_predvalues1_1.put(person1+","+person2, probability)
				persons1_persons2_predvalues1_1.put(person2+","+person1, probability)
			}
			
		}
	}
	System.out.println("[DEBUG]: End GreedyOneToOneMatching: countOfMatches = " + countOfMatches);
	
	return persons1_persons2_predvalues1_1;
}

//this function takes as input the structure persons1_persons2_predvalues (e.g. <"id1,id2",value>)
//and produces a structure of the form TreeMultimap<Double, String> (e.g. <value,"id1,id2">)
//in descreasing value order
TreeMultimap<Double, String> parsePredictions(
	HashMap<String, String> persons1_persons2_predvalues, Double threshold){

	//to store the values in reverse order
	final Comparator<Double> DECREASING_DOUBLE_COMPARATOR = Ordering.<Double>natural().reverse().nullsFirst();
	
	//the following structure keeps instances of the form <sim_value,"person1,person2">
	//the structure is SORTED with respect to the similarity (probability)
	TreeMultimap<Double, String> sortedProbabilityPersonsMatched  = new TreeMultimap<>(DECREASING_DOUBLE_COMPARATOR, Ordering.natural());
	
	//iterate over the structure HashMap<String, Double> persons1_persons2_predvalues
	//and fill in the new structure sortedProbabilityPersonsMatched
	for(String person1_person2 : persons1_persons2_predvalues.keySet()){
		//insert the entry only if the values are greater or equal than the threshold
		double predValue = persons1_persons2_predvalues.get(person1_person2);
		if(predValue>=threshold)
			sortedProbabilityPersonsMatched.put(predValue, person1_person2);
	}
	
	System.out.println("[DEBUG]: Just created the structure with the predictions sorted in reverse order");
	System.out.println("[DEBUG]: Size of the structure = " + sortedProbabilityPersonsMatched.size());
	
	return sortedProbabilityPersonsMatched;
}

//Compute confusion matrix after the 1-1 matching
ConfusionMatrix computeConfusionMatrix1_1(HashMap<String, Double> persons1_persons2_predvalues1_1,
		HashMap<String, Double> persons1_persons2_matchvalues){
		
		ConfusionMatrix confMat = new ConfusionMatrix();
		
		println "Compute the confusion matrix after the 1-1 matching algorithm"
		double match_true_value, match_predicted_value;
		int n=0;
		//instead of iterating over the structure persons1_persons2_predvalues1_1
		//as in the other computeConfusionMatrix function, we now iterate over
		//the structure persons1_persons2_matchvalues
		Set<String> persons = persons1_persons2_matchvalues.keySet();
		for (String person1_person2 : persons){
			//get the true value
			match_true_value = persons1_persons2_matchvalues.get(person1_person2);
			
			//just for debugging also test for the reverse pair
			String[] reverse_persons = person1_person2.split(",");
			String person1 = reverse_persons[0];
			String person2 = reverse_persons[1];
			String person2_person1 = person2+","+person1
			
			if(persons1_persons2_predvalues1_1.containsKey(person1_person2) ||
				persons1_persons2_predvalues1_1.containsKey(person2_person1)){
				
				if(match_true_value==1)
					confMat.classYes_predictedYes++;
				else
					confMat.classNo_predictedYes++;
			}
			else{
				if(match_true_value==1)
					confMat.classYes_predictedNo++;
				else
					confMat.classNo_predictedNo++;
			}
			
		}
		return confMat;
		
}
			
	
	


PSLModel createModel(Object object, DataStore data, 
	int posFirstNameWeight, int posLastNameWeight, int posMaidenNameWeight,  
	double negFirstNameWeight, double negLastNameWeight, double negMaidenNameWeight, 
	int ageWeight, int negAgeWeight, int negGenderLivingWeight,
	double firstDegreeSimWeight, double secDegreeSimWeight,
	double firstDegreeTransWeight, double secDegreeTransWeight, double negFirstDegreeTransWeight, 
	double bijectionWeight, double transWeight, double priorWeight,
	boolean persInfRun, boolean firstDegSimRun, 
	boolean secDegSimRun, boolean firstDegTranRun, boolean secDegTranRun, 
	boolean bijRun, boolean transRun){
	
	PSLModel m = new PSLModel(object, data)
	
	//definition of predicates
	
	//general predicates
	m.add predicate: "person", types: [ArgumentType.UniqueID]
	m.add predicate: "belongsToFamily", types:[ArgumentType.UniqueID,ArgumentType.UniqueID]
	m.add predicate: "belongsToParticipant", types:[ArgumentType.UniqueID,ArgumentType.UniqueID]	
	
	//similarity predicates
	m.add predicate: "firstNameJaroSim", types:[ArgumentType.UniqueID,ArgumentType.UniqueID]
	m.add predicate: "maidenNameJaroSim", types:[ArgumentType.UniqueID,ArgumentType.UniqueID]
	m.add predicate: "lastNameJaroSim", types:[ArgumentType.UniqueID,ArgumentType.UniqueID]
	
	//this predicate is used for the transitive relational rules
	m.add predicate: "maxJaroLevFirstName", types:[ArgumentType.UniqueID,ArgumentType.UniqueID]
	
	m.add predicate: "firstNameLevSim", types:[ArgumentType.UniqueID,ArgumentType.UniqueID]
	m.add predicate: "maidenNameLevSim", types:[ArgumentType.UniqueID,ArgumentType.UniqueID]
	m.add predicate: "lastNameLevSim", types:[ArgumentType.UniqueID,ArgumentType.UniqueID]
	
	m.add predicate: "similarAge", types:[ArgumentType.UniqueID,ArgumentType.UniqueID]
	m.add predicate: "sameGender", types:[ArgumentType.UniqueID,ArgumentType.UniqueID]
	m.add predicate: "sameLivingStatus", types:[ArgumentType.UniqueID,ArgumentType.UniqueID]
	
	//relationship similarities equivalent to logistic regression ones
	//first degree individual
	m.add predicate: "motherSim", types:[ArgumentType.UniqueID, ArgumentType.UniqueID]	//total
	m.add predicate: "fatherSim", types:[ArgumentType.UniqueID, ArgumentType.UniqueID]	//total
	m.add predicate: "brothersSim", types:[ArgumentType.UniqueID, ArgumentType.UniqueID]	//total
	m.add predicate: "sistersSim", types:[ArgumentType.UniqueID, ArgumentType.UniqueID]	//total
	m.add predicate: "daughtersSim", types:[ArgumentType.UniqueID, ArgumentType.UniqueID]	//total
	m.add predicate: "sonsSim", types:[ArgumentType.UniqueID, ArgumentType.UniqueID]		//total
	m.add predicate: "spouseSim", types:[ArgumentType.UniqueID, ArgumentType.UniqueID]	//total
	
	//second degree individual
	m.add predicate: "grandMothersSim", types:[ArgumentType.UniqueID, ArgumentType.UniqueID]	//total
	m.add predicate: "grandFathersSim", types:[ArgumentType.UniqueID, ArgumentType.UniqueID]	//total
	m.add predicate: "grandDaughtersSim", types:[ArgumentType.UniqueID, ArgumentType.UniqueID]	//total
	m.add predicate: "grandSonsSim", types:[ArgumentType.UniqueID, ArgumentType.UniqueID]	//total
	m.add predicate: "unclesSim", types:[ArgumentType.UniqueID, ArgumentType.UniqueID]	//total
	m.add predicate: "auntsSim", types:[ArgumentType.UniqueID, ArgumentType.UniqueID]	//total
	m.add predicate: "niecesSim", types:[ArgumentType.UniqueID, ArgumentType.UniqueID]	//total
	m.add predicate: "nephewsSim", types:[ArgumentType.UniqueID, ArgumentType.UniqueID]	//total
	
	//predicates for 1st degree familial relations - these predicates are for the transitivity like rules
	m.add predicate: "hasMother", types:[ArgumentType.UniqueID, ArgumentType.UniqueID]	//total 5,087
	m.add predicate: "hasFather", types:[ArgumentType.UniqueID, ArgumentType.UniqueID]	//total 5,050
	m.add predicate: "hasBrother", types:[ArgumentType.UniqueID, ArgumentType.UniqueID]	//total 13,904
	m.add predicate: "hasSister", types:[ArgumentType.UniqueID, ArgumentType.UniqueID]	//total 13,121
	m.add predicate: "hasDaughter", types:[ArgumentType.UniqueID, ArgumentType.UniqueID]	//total 4,051
	m.add predicate: "hasSon", types:[ArgumentType.UniqueID, ArgumentType.UniqueID]	//total 4,118
	m.add predicate: "hasSpouse", types:[ArgumentType.UniqueID, ArgumentType.UniqueID]	//total 1,652
	
	//predicates for 2nd degree familial relations
	m.add predicate: "hasGrandMother", types:[ArgumentType.UniqueID, ArgumentType.UniqueID]	//total
	m.add predicate: "hasGrandFather", types:[ArgumentType.UniqueID, ArgumentType.UniqueID]	//total
	m.add predicate: "hasGrandDaughter", types:[ArgumentType.UniqueID, ArgumentType.UniqueID]	//total
	m.add predicate: "hasGrandSon", types:[ArgumentType.UniqueID, ArgumentType.UniqueID]	//total
	m.add predicate: "hasUncle", types:[ArgumentType.UniqueID, ArgumentType.UniqueID]	//total
	m.add predicate: "hasAunt", types:[ArgumentType.UniqueID, ArgumentType.UniqueID]	//total
	m.add predicate: "hasNiece", types:[ArgumentType.UniqueID, ArgumentType.UniqueID]	//total
	m.add predicate: "hasNephew", types:[ArgumentType.UniqueID, ArgumentType.UniqueID]	//total

	//this predicate is always observed and is used only for blocking!
	m.add predicate: "samePersonBlock", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
	//the predicate we are interested in predicting
	m.add predicate: "samePerson", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
	
	
	//bijection rule
	if(bijRun)
		m.add rule : ( person(A) & person(B) & person(C) & samePerson(A,B) 
		& samePersonBlock(A,B) & samePersonBlock(A,C)
		& belongsToFamily(A,W) & belongsToFamily(B,W) & belongsToFamily(C,W)	//they should belong to the same family
		& belongsToParticipant(A,M) & belongsToParticipant(B,N) & belongsToParticipant(C,N)
		& (M-N)	& (B-C)			//M and N should be distinct constants
		) >> ~samePerson(A,C),  squared: sqPotentials, weight : bijectionWeight
	
	
	//transitivity rule
	if(transRun)		
	 m.add rule : ( person(A) & person(B) & person(C) & sameGender(A,C)
		& samePerson(A,B) & samePerson(B,C)
		& samePersonBlock(A,B) & samePersonBlock(B,C) & samePersonBlock(A,C)
		& belongsToFamily(A,W) & belongsToFamily(B,W) & belongsToFamily(C,W)	//they should belong to the same family
		& belongsToParticipant(A,M) & belongsToParticipant(B,N) & belongsToParticipant(C,O)//A and C should belong to different tree
		& (M-N) & (N-O) & (M-O)	& (B-C)	& (A-C)//M and N and O should be distinct constants
		) >> samePerson(A,C),  squared: sqPotentials, weight : transWeight
	
	//NAME SIMILARITIES
	//Jaro Similarity
	m.add rule : ( person(A) & person(B) & samePersonBlock(A,B) & firstNameJaroSim(A,B) & sameGender(A,B)
				&  belongsToFamily(A,W) & belongsToFamily(B,W)
				& belongsToParticipant(A,M) & belongsToParticipant(B,N)
				) >> samePerson(A,B),  squared: sqPotentials, weight : posFirstNameWeight

	
	m.add rule : ( person(A) & person(B) & samePersonBlock(A,B) &  maidenNameJaroSim(A,B) & sameGender(A,B)
				&  belongsToFamily(A,W) & belongsToFamily(B,W)
				& belongsToParticipant(A,M) & belongsToParticipant(B,N)
				) >> samePerson(A,B),  squared: sqPotentials, weight : posMaidenNameWeight
	
			
	m.add rule : ( person(A) & person(B) & samePersonBlock(A,B) & lastNameJaroSim(A,B) & sameGender(A,B)
				&  belongsToFamily(A,W) & belongsToFamily(B,W)
				& belongsToParticipant(A,M) & belongsToParticipant(B,N)
				) >> samePerson(A,B),  squared: sqPotentials, weight : posLastNameWeight

	//introduce the inverse rules!
	m.add rule : ( person(A) & person(B) & samePersonBlock(A,B) & ~firstNameJaroSim(A,B)
				&  belongsToFamily(A,W) & belongsToFamily(B,W)
				& belongsToParticipant(A,M) & belongsToParticipant(B,N)
				) >> ~samePerson(A,B),  squared: sqPotentials, weight : negFirstNameWeight

			
	m.add rule : ( person(A) & person(B) & samePersonBlock(A,B) &  ~maidenNameJaroSim(A,B)
				&  belongsToFamily(A,W) & belongsToFamily(B,W)
				& belongsToParticipant(A,M) & belongsToParticipant(B,N)
				) >> ~samePerson(A,B),  squared: sqPotentials, weight : negMaidenNameWeight
	
			
	m.add rule : ( person(A) & person(B) & samePersonBlock(A,B) & ~lastNameJaroSim(A,B)
				&  belongsToFamily(A,W) & belongsToFamily(B,W)
				& belongsToParticipant(A,M) & belongsToParticipant(B,N)
				) >> ~samePerson(A,B),  squared: sqPotentials, weight : negLastNameWeight
		
					
	//Levenshtein Similarity
	m.add rule : ( person(A) & person(B) & samePersonBlock(A,B) & firstNameLevSim(A,B) & sameGender(A,B)
				&  belongsToFamily(A,W) & belongsToFamily(B,W)
				& belongsToParticipant(A,M) & belongsToParticipant(B,N)
				) >> samePerson(A,B),  squared: sqPotentials, weight : posFirstNameWeight
	
	
	m.add rule : ( person(A) & person(B) & samePersonBlock(A,B)  & maidenNameLevSim(A,B) & sameGender(A,B)
				&  belongsToFamily(A,W) & belongsToFamily(B,W)
				& belongsToParticipant(A,M) & belongsToParticipant(B,N)
				) >> samePerson(A,B),  squared: sqPotentials, weight : posMaidenNameWeight
	
			
	m.add rule : ( person(A) & person(B) & samePersonBlock(A,B) & lastNameLevSim(A,B) & sameGender(A,B)
				&  belongsToFamily(A,W) & belongsToFamily(B,W)
				& belongsToParticipant(A,M) & belongsToParticipant(B,N)
				) >> samePerson(A,B),  squared: sqPotentials, weight : posLastNameWeight

	//introduce the reverse rules
	m.add rule : ( person(A) & person(B) & samePersonBlock(A,B) & ~firstNameLevSim(A,B)
				&  belongsToFamily(A,W) & belongsToFamily(B,W)
				& belongsToParticipant(A,M) & belongsToParticipant(B,N)
				) >> ~samePerson(A,B),  squared: sqPotentials, weight : negFirstNameWeight
	
		
	m.add rule : ( person(A) & person(B) & samePersonBlock(A,B) & ~maidenNameLevSim(A,B)
				&  belongsToFamily(A,W) & belongsToFamily(B,W)
				& belongsToParticipant(A,M) & belongsToParticipant(B,N)
				) >> ~samePerson(A,B),  squared: sqPotentials, weight : negMaidenNameWeight
	
			
	m.add rule : ( person(A) & person(B) & samePersonBlock(A,B) & ~lastNameLevSim(A,B)
				&  belongsToFamily(A,W) & belongsToFamily(B,W)
				& belongsToParticipant(A,M) & belongsToParticipant(B,N)
				) >> ~samePerson(A,B),  squared: sqPotentials, weight : negLastNameWeight
	
	if(persInfRun){		
	//PERSONAL INFORMATION SIMILARITIES
	//similar age
	m.add rule : ( person(A) & person(B)  & samePersonBlock(A,B) & similarAge(A,B)
				&  belongsToFamily(A,W) & belongsToFamily(B,W)
				& belongsToParticipant(A,M) & belongsToParticipant(B,N)
				) >> samePerson(A,B),  squared: sqPotentials, weight : ageWeight
	
		
	m.add rule : ( person(A) & person(B)  & samePersonBlock(A,B) & ~similarAge(A,B)
				&  belongsToFamily(A,W) & belongsToFamily(B,W)
				& belongsToParticipant(A,M) & belongsToParticipant(B,N)
				) >> ~samePerson(A,B),  squared: sqPotentials, weight : negAgeWeight
		
			
	m.add rule : ( person(A) & person(B)  & samePersonBlock(A,B) & ~sameGender(A,B)
				&  belongsToFamily(A,W) & belongsToFamily(B,W)
				& belongsToParticipant(A,M) & belongsToParticipant(B,N)
				) >> ~samePerson(A,B),  squared: oneDirPotentials, weight : negGenderLivingWeight
		
	m.add rule : ( person(A) & person(B)  & samePersonBlock(A,B) & ~sameLivingStatus(A,B)
				&  belongsToFamily(A,W) & belongsToFamily(B,W)
				& belongsToParticipant(A,M) & belongsToParticipant(B,N)
				) >> ~samePerson(A,B),  squared: oneDirPotentials, weight : negGenderLivingWeight
	
							
	}
			
	//RELATIONSHIP INFORMATION RULES
	//INDIVIDUAL SIMILARITIES
	if(firstDegSimRun){		
	//first degree relationships
	m.add rule : ( person(A) & person(B)  & samePersonBlock(A,B) & motherSim(A,B) & sameGender(A,B)
				&  belongsToFamily(A,W) & belongsToFamily(B,W)
				& belongsToParticipant(A,M) & belongsToParticipant(B,N)
				) >> samePerson(A,B),  squared: sqPotentials, weight : firstDegreeSimWeight
	
	m.add rule : ( person(A) & person(B)  & samePersonBlock(A,B) & fatherSim(A,B) & sameGender(A,B)
				&  belongsToFamily(A,W) & belongsToFamily(B,W)
				& belongsToParticipant(A,M) & belongsToParticipant(B,N)
				) >> samePerson(A,B),  squared: sqPotentials, weight : firstDegreeSimWeight
			
	m.add rule : ( person(A) & person(B)  & samePersonBlock(A,B) & daughtersSim(A,B) & sameGender(A,B)
				&  belongsToFamily(A,W) & belongsToFamily(B,W)
				& belongsToParticipant(A,M) & belongsToParticipant(B,N)
				) >> samePerson(A,B),  squared: sqPotentials, weight : firstDegreeSimWeight
			
	m.add rule : ( person(A) & person(B)  & samePersonBlock(A,B) & sonsSim(A,B) & sameGender(A,B)
				&  belongsToFamily(A,W) & belongsToFamily(B,W)
				& belongsToParticipant(A,M) & belongsToParticipant(B,N)
				) >> samePerson(A,B),  squared: sqPotentials, weight : firstDegreeSimWeight
			
	m.add rule : ( person(A) & person(B)  & samePersonBlock(A,B) & sistersSim(A,B) & sameGender(A,B)
				&  belongsToFamily(A,W) & belongsToFamily(B,W)
				& belongsToParticipant(A,M) & belongsToParticipant(B,N)
				) >> samePerson(A,B),  squared: sqPotentials, weight : firstDegreeSimWeight
			
	m.add rule : ( person(A) & person(B)  & samePersonBlock(A,B) & brothersSim(A,B) & sameGender(A,B)
				&  belongsToFamily(A,W) & belongsToFamily(B,W)
				& belongsToParticipant(A,M) & belongsToParticipant(B,N)
				) >> samePerson(A,B),  squared: sqPotentials, weight : firstDegreeSimWeight
			
	m.add rule : ( person(A) & person(B)  & samePersonBlock(A,B) & spouseSim(A,B) & sameGender(A,B)
				&  belongsToFamily(A,W) & belongsToFamily(B,W)
				& belongsToParticipant(A,M) & belongsToParticipant(B,N)
				) >> samePerson(A,B),  squared: sqPotentials, weight : firstDegreeSimWeight
	
	m.add rule : ( person(A) & person(B)  & samePersonBlock(A,B) & ~spouseSim(A,B) & sameGender(A,B)
				&  belongsToFamily(A,W) & belongsToFamily(B,W)
				& belongsToParticipant(A,M) & belongsToParticipant(B,N)
				) >> ~samePerson(A,B),  squared: sqPotentials, weight : firstDegreeSimWeight
	
	m.add rule : ( person(A) & person(B)  & samePersonBlock(A,B) & ~motherSim(A,B) & sameGender(A,B)
				&  belongsToFamily(A,W) & belongsToFamily(B,W)
				& belongsToParticipant(A,M) & belongsToParticipant(B,N)
				) >> ~samePerson(A,B),  squared: sqPotentials, weight : firstDegreeSimWeight
	
	m.add rule : ( person(A) & person(B)  & samePersonBlock(A,B) & ~fatherSim(A,B) & sameGender(A,B)
				&  belongsToFamily(A,W) & belongsToFamily(B,W)
				& belongsToParticipant(A,M) & belongsToParticipant(B,N)
				) >> ~samePerson(A,B),  squared: sqPotentials, weight : firstDegreeSimWeight
	
	
	}		
			
		
	
	if(secDegSimRun){
	//second degree relationships
	m.add rule : ( person(A) & person(B)  & samePersonBlock(A,B) & grandMothersSim(A,B) & sameGender(A,B)
				&  belongsToFamily(A,W) & belongsToFamily(B,W)
				& belongsToParticipant(A,M) & belongsToParticipant(B,N)
				) >> samePerson(A,B),  squared: sqPotentials, weight : secDegreeSimWeight
			
	m.add rule : ( person(A) & person(B)  & samePersonBlock(A,B) & grandFathersSim(A,B) & sameGender(A,B)
				&  belongsToFamily(A,W) & belongsToFamily(B,W)
				& belongsToParticipant(A,M) & belongsToParticipant(B,N)
				) >> samePerson(A,B),  squared: sqPotentials, weight : secDegreeSimWeight
					
	m.add rule : ( person(A) & person(B)  & samePersonBlock(A,B) & grandDaughtersSim(A,B) & sameGender(A,B)
				&  belongsToFamily(A,W) & belongsToFamily(B,W)
				& belongsToParticipant(A,M) & belongsToParticipant(B,N)
				) >> samePerson(A,B),  squared: sqPotentials, weight : secDegreeSimWeight
					
	m.add rule : ( person(A) & person(B)  & samePersonBlock(A,B) & grandSonsSim(A,B) & sameGender(A,B)
				&  belongsToFamily(A,W) & belongsToFamily(B,W)
				& belongsToParticipant(A,M) & belongsToParticipant(B,N)
				) >> samePerson(A,B),  squared: sqPotentials, weight : secDegreeSimWeight
					
	m.add rule : ( person(A) & person(B)  & samePersonBlock(A,B) & auntsSim(A,B) & sameGender(A,B)
				&  belongsToFamily(A,W) & belongsToFamily(B,W)
				& belongsToParticipant(A,M) & belongsToParticipant(B,N)
				) >> samePerson(A,B),  squared: sqPotentials, weight : secDegreeSimWeight
					
	m.add rule : ( person(A) & person(B)  & samePersonBlock(A,B) & unclesSim(A,B) & sameGender(A,B)
				&  belongsToFamily(A,W) & belongsToFamily(B,W)
				& belongsToParticipant(A,M) & belongsToParticipant(B,N)
				) >> samePerson(A,B),  squared: sqPotentials, weight : secDegreeSimWeight
					
	m.add rule : ( person(A) & person(B)  & samePersonBlock(A,B) & niecesSim(A,B) & sameGender(A,B)
				&  belongsToFamily(A,W) & belongsToFamily(B,W)
				& belongsToParticipant(A,M) & belongsToParticipant(B,N)
				) >> samePerson(A,B),  squared: sqPotentials, weight : secDegreeSimWeight
			
	m.add rule : ( person(A) & person(B)  & samePersonBlock(A,B) & nephewsSim(A,B) & sameGender(A,B)
				&  belongsToFamily(A,W) & belongsToFamily(B,W)
				& belongsToParticipant(A,M) & belongsToParticipant(B,N)
				) >> samePerson(A,B),  squared: sqPotentials, weight : secDegreeSimWeight
	}
						
	//transitivity rules using 1st degree familial relationships
	//e.g. if two persons that both have a sister are the same,
	//and the names of those sisters are close enough then the sisters may be the same persons as well
	//we check the first name only, the last name is expected to be the same for all sisters
	
	if(firstDegTranRun){
	//mother
	m.add rule : ( person(P1) & person(P2) & person(S1) & person(S2) & sameGender(S1,S2)
				& samePersonBlock(P1,P2) & samePersonBlock(S1,S2) & samePerson(P1,P2)
				& hasMother(P1,S1) & hasMother(P2,S2) 
				& belongsToFamily(P1,F) & belongsToFamily(P2,F) & belongsToFamily(S1,F) & belongsToFamily(S2,F)	//they should belong to the same family
				& belongsToParticipant(P1,M) & belongsToParticipant(S1,M)
				& belongsToParticipant(P2,N) & belongsToParticipant(S2,N) //A and C should belong to different tree
				) >> samePerson(S1,S2),  squared: oneDirPotentials, weight : firstDegreeTransWeight
			
	//father
	m.add rule : ( person(P1) & person(P2) & person(S1) & person(S2) & sameGender(S1,S2)
				& samePersonBlock(P1,P2) & samePersonBlock(S1,S2) & samePerson(P1,P2)
				& hasFather(P1,S1) & hasFather(P2,S2) 
				& belongsToFamily(P1,F) & belongsToFamily(P2,F) & belongsToFamily(S1,F) & belongsToFamily(S2,F)	//they should belong to the same family
				& belongsToParticipant(P1,M) & belongsToParticipant(S1,M)
				& belongsToParticipant(P2,N) & belongsToParticipant(S2,N) //A and C should belong to different tree
				) >> samePerson(S1,S2),  squared: oneDirPotentials, weight : firstDegreeTransWeight
		
					
	//hasSpouse
	m.add rule : ( person(P1) & person(P2) & person(S1) & person(S2) & sameGender(S1,S2)
				& samePersonBlock(P1,P2) & samePersonBlock(S1,S2) & samePerson(P1,P2)
				& hasSpouse(P1,S1) & hasSpouse(P2,S2) 
				& belongsToFamily(P1,F) & belongsToFamily(P2,F) & belongsToFamily(S1,F) & belongsToFamily(S2,F)	//they should belong to the same family
				& belongsToParticipant(P1,M) & belongsToParticipant(S1,M)
				& belongsToParticipant(P2,N) & belongsToParticipant(S2,N) //A and C should belong to different tree
				) >> samePerson(S1,S2),  squared: sqPotentials, weight : firstDegreeTransWeight

	//negative for spouse 
	m.add rule : ( person(P1) & person(P2) & person(S1) & person(S2) & sameGender(S1,S2)
		& samePersonBlock(P1,P2) & samePersonBlock(S1,S2) & ~samePerson(P1,P2)
		& hasSpouse(P1,S1) & hasSpouse(P2,S2) 
		& belongsToFamily(P1,F) & belongsToFamily(P2,F) & belongsToFamily(S1,F) & belongsToFamily(S2,F)	//they should belong to the same family
		& belongsToParticipant(P1,M) & belongsToParticipant(S1,M)
		& belongsToParticipant(P2,N) & belongsToParticipant(S2,N) //A and C should belong to different tree
		) >> ~samePerson(S1,S2),  squared: sqPotentials, weight : negFirstDegreeTransWeight
			
		//sister
		m.add rule : ( person(P1) & person(P2) & person(S1) & person(S2) & sameGender(S1,S2)
					& samePersonBlock(P1,P2) & samePersonBlock(S1,S2) & samePerson(P1,P2)
					& hasSister(P1,S1) & hasSister(P2,S2) & maxJaroLevFirstName(S1,S2)
					& belongsToFamily(P1,F) & belongsToFamily(P2,F) & belongsToFamily(S1,F) & belongsToFamily(S2,F)	//they should belong to the same family
					& belongsToParticipant(P1,M) & belongsToParticipant(S1,M)
					& belongsToParticipant(P2,N) & belongsToParticipant(S2,N) //A and C should belong to different tree
					) >> samePerson(S1,S2),  squared: oneDirPotentials, weight : firstDegreeTransWeight
		
		//brother
		m.add rule : ( person(P1) & person(P2) & person(S1) & person(S2) & sameGender(S1,S2)
					& samePersonBlock(P1,P2) & samePersonBlock(S1,S2) & samePerson(P1,P2)
					& hasBrother(P1,S1) & hasBrother(P2,S2) & maxJaroLevFirstName(S1,S2)
					& belongsToFamily(P1,F) & belongsToFamily(P2,F) & belongsToFamily(S1,F) & belongsToFamily(S2,F)	//they should belong to the same family
					& belongsToParticipant(P1,M) & belongsToParticipant(S1,M)
					& belongsToParticipant(P2,N) & belongsToParticipant(S2,N) //A and C should belong to different tree
					) >> samePerson(S1,S2),  squared: oneDirPotentials, weight : firstDegreeTransWeight
		
	//daughter
	m.add rule : ( person(P1) & person(P2) & person(S1) & person(S2) & sameGender(S1,S2)
				& samePersonBlock(P1,P2) & samePersonBlock(S1,S2) & samePerson(P1,P2)
				& hasDaughter(P1,S1) & hasDaughter(P2,S2) & maxJaroLevFirstName(S1,S2)
				& belongsToFamily(P1,F) & belongsToFamily(P2,F) & belongsToFamily(S1,F) & belongsToFamily(S2,F)	//they should belong to the same family
				& belongsToParticipant(P1,M) & belongsToParticipant(S1,M)
				& belongsToParticipant(P2,N) & belongsToParticipant(S2,N) //A and C should belong to different tree
				) >> samePerson(S1,S2),  squared: oneDirPotentials, weight : firstDegreeTransWeight
	
	//son
	m.add rule : ( person(P1) & person(P2) & person(S1) & person(S2) & sameGender(S1,S2)
				& samePersonBlock(P1,P2) & samePersonBlock(S1,S2) & samePerson(P1,P2)
				& hasSon(P1,S1) & hasSon(P2,S2) & maxJaroLevFirstName(S1,S2)
				& belongsToFamily(P1,F) & belongsToFamily(P2,F) & belongsToFamily(S1,F) & belongsToFamily(S2,F)	//they should belong to the same family
				& belongsToParticipant(P1,M) & belongsToParticipant(S1,M)
				& belongsToParticipant(P2,N) & belongsToParticipant(S2,N) //A and C should belong to different tree
				) >> samePerson(S1,S2),  squared: oneDirPotentials, weight : firstDegreeTransWeight
	}
			
			
	
	if(secDegTranRun){
	//same rules but for 2nd degree relations
	//grandmother
	m.add rule : ( person(P1) & person(P2) & person(S1) & person(S2) & sameGender(S1,S2)
				& samePersonBlock(P1,P2) & samePersonBlock(S1,S2) & samePerson(P1,P2)
				& hasGrandMother(P1,S1) & hasGrandMother(P2,S2) & maxJaroLevFirstName(S1,S2) 
				& belongsToFamily(P1,F) & belongsToFamily(P2,F) & belongsToFamily(S1,F) & belongsToFamily(S2,F)	//they should belong to the same family
				& belongsToParticipant(P1,M) & belongsToParticipant(S1,M)
				& belongsToParticipant(P2,N) & belongsToParticipant(S2,N) //A and C should belong to different tree
				) >> samePerson(S1,S2),  squared: sqPotentials, weight : secDegreeTransWeight
			
	//grandfather
	m.add rule : ( person(P1) & person(P2) & person(S1) & person(S2) & sameGender(S1,S2)
				& samePersonBlock(P1,P2) & samePersonBlock(S1,S2) & samePerson(P1,P2)
				& hasGrandFather(P1,S1) & hasGrandFather(P2,S2) & maxJaroLevFirstName(S1,S2) 
				& belongsToFamily(P1,F) & belongsToFamily(P2,F) & belongsToFamily(S1,F) & belongsToFamily(S2,F)	//they should belong to the same family
				& belongsToParticipant(P1,M) & belongsToParticipant(S1,M)
				& belongsToParticipant(P2,N) & belongsToParticipant(S2,N) //A and C should belong to different tree
				) >> samePerson(S1,S2),  squared: sqPotentials, weight : secDegreeTransWeight
			
	//granddaughter
	m.add rule : ( person(P1) & person(P2) & person(S1) & person(S2) & sameGender(S1,S2)
				& samePersonBlock(P1,P2) & samePersonBlock(S1,S2) & samePerson(P1,P2)
				& hasGrandDaughter(P1,S1) & hasGrandDaughter(P2,S2) & maxJaroLevFirstName(S1,S2) 
				& belongsToFamily(P1,F) & belongsToFamily(P2,F) & belongsToFamily(S1,F) & belongsToFamily(S2,F)	//they should belong to the same family
				& belongsToParticipant(P1,M) & belongsToParticipant(S1,M)
				& belongsToParticipant(P2,N) & belongsToParticipant(S2,N)//A and C should belong to different tree
				) >> samePerson(S1,S2),  squared: sqPotentials, weight : secDegreeTransWeight
			
	//grandSon
	m.add rule : ( person(P1) & person(P2) & person(S1) & person(S2) & sameGender(S1,S2)
				& samePersonBlock(P1,P2) & samePersonBlock(S1,S2) & samePerson(P1,P2)
				& hasGrandSon(P1,S1) & hasGrandSon(P2,S2) & maxJaroLevFirstName(S1,S2) 
				& belongsToFamily(P1,F) & belongsToFamily(P2,F) & belongsToFamily(S1,F) & belongsToFamily(S2,F)	//they should belong to the same family
				& belongsToParticipant(P1,M) & belongsToParticipant(S1,M)
				& belongsToParticipant(P2,N) & belongsToParticipant(S2,N) //A and C should belong to different tree
				) >> samePerson(S1,S2),  squared: sqPotentials, weight : secDegreeTransWeight
			
	//aunt
	m.add rule : ( person(P1) & person(P2) & person(S1) & person(S2) & sameGender(S1,S2)
				& samePersonBlock(P1,P2) & samePersonBlock(S1,S2) & samePerson(P1,P2)
				& hasAunt(P1,S1) & hasAunt(P2,S2) & maxJaroLevFirstName(S1,S2) 
				& belongsToFamily(P1,F) & belongsToFamily(P2,F) & belongsToFamily(S1,F) & belongsToFamily(S2,F)	//they should belong to the same family
				& belongsToParticipant(P1,M) & belongsToParticipant(S1,M)
				& belongsToParticipant(P2,N) & belongsToParticipant(S2,N) //A and C should belong to different tree
				) >> samePerson(S1,S2),  squared: sqPotentials, weight : secDegreeTransWeight
			
	//uncle
	m.add rule : ( person(P1) & person(P2) & person(S1) & person(S2) & sameGender(S1,S2)
				& samePersonBlock(P1,P2) & samePersonBlock(S1,S2) & samePerson(P1,P2)
				& hasUncle(P1,S1) & hasUncle(P2,S2) & maxJaroLevFirstName(S1,S2) 
				& belongsToFamily(P1,F) & belongsToFamily(P2,F) & belongsToFamily(S1,F) & belongsToFamily(S2,F)	//they should belong to the same family
				& belongsToParticipant(P1,M) & belongsToParticipant(S1,M)
				& belongsToParticipant(P2,N) & belongsToParticipant(S2,N) //A and C should belong to different tree
				) >> samePerson(S1,S2),  squared: sqPotentials, weight : secDegreeTransWeight
			
	//niece
	m.add rule : ( person(P1) & person(P2) & person(S1) & person(S2) & sameGender(S1,S2)
				& samePersonBlock(P1,P2) & samePersonBlock(S1,S2) & samePerson(P1,P2)
				& hasNiece(P1,S1) & hasNiece(P2,S2) & maxJaroLevFirstName(S1,S2) 
				& belongsToFamily(P1,F) & belongsToFamily(P2,F) & belongsToFamily(S1,F) & belongsToFamily(S2,F)	//they should belong to the same family
				& belongsToParticipant(P1,M) & belongsToParticipant(S1,M)
				& belongsToParticipant(P2,N) & belongsToParticipant(S2,N)
				) >> samePerson(S1,S2),  squared: sqPotentials, weight : secDegreeTransWeight
	
	//nephew
	m.add rule : ( person(P1) & person(P2) & person(S1) & person(S2) & sameGender(S1,S2)
				& samePersonBlock(P1,P2) & samePersonBlock(S1,S2) & samePerson(P1,P2)
				& hasNephew(P1,S1) & hasNephew(P2,S2) & maxJaroLevFirstName(S1,S2) 
				& belongsToFamily(P1,F) & belongsToFamily(P2,F) & belongsToFamily(S1,F) & belongsToFamily(S2,F)	//they should belong to the same family
				& belongsToParticipant(P1,M) & belongsToParticipant(S1,M)
				& belongsToParticipant(P2,N) & belongsToParticipant(S2,N)
				) >> samePerson(S1,S2),  squared: sqPotentials, weight : secDegreeTransWeight
	}
	
	//prior
	m.add rule: ~samePerson(A,B), squared: sqPotentials, weight: priorWeight
	
	
return m;
	
}

//function that performs weight learning
void performWeightLearning(PSLModel m, DataStore data, ConfigBundle config, 
	Partition evidencePartitionCommon, Partition targetPartitionWL, 
	Partition evidencePartitionWL, Partition trueDataPartitionWL){
	
	//dbWL is the db just for the WL
	Database dbWL = data.getDatabase(targetPartitionWL,
	[person, samePersonBlock, belongsToFamily, belongsToParticipant,
		similarAge, sameGender, sameLivingStatus,
		firstNameJaroSim, maidenNameJaroSim, lastNameJaroSim, 
		firstNameLevSim, maidenNameLevSim, lastNameLevSim,
		maxJaroLevFirstName,
		motherSim, fatherSim, daughtersSim, sonsSim, sistersSim, brothersSim, spouseSim,//first degree
		grandMothersSim, grandFathersSim, grandDaughtersSim, grandSonsSim, auntsSim, unclesSim, niecesSim, nephewsSim, //second degree
		hasMother, hasFather, hasSister, hasBrother, hasSister, hasDaughter, hasSon, hasSpouse,
		hasGrandMother, hasGrandFather, hasGrandDaughter, hasGrandSon, hasUncle, hasAunt, hasNephew, hasNiece
	] as Set, evidencePartitionCommon, evidencePartitionWL);
	
	println "TIME before weight learning is " + (new Date())
	Database trueDataDB = data.getDatabase(trueDataPartitionWL, [samePerson] as Set);
	
	MaxLikelihoodMPE weightLearning = new MaxLikelihoodMPE(m, dbWL, trueDataDB, config);
	
	weightLearning.learn();
	weightLearning.close();
	
	//print the new model
	println ""
	println "Learned model:"
	println m
	dbWL.close();	//close this db as we will not use it again
	
	println "TIME after weight learning " + new Date();
}


//load the ground truth data
//the matches object updates the number of true matches and the number of true non matches
//the format of the file is "personID1	personID2	match/nomatch"
HashMap<String, Double>  loadGroundTruth(String fileName, Matches matches){
	
	HashMap<String, Double> persons1_persons2_matchvalues = new HashMap();
	
	def labels = new File(fileName)
	def words, person1, person2, match_value
	labels.eachLine {
		line ->
		words = line.split("\t")
		person1=words[0].toString();
		person2=words[1].toString();
		match_value=words[2].toDouble();
		if(match_value==1)
			matches.numberOfTrueMatches++;
		else
			matches.numberOfTrueNonMatches++;
			
		persons1_persons2_matchvalues.put(person1+","+person2, match_value);
		
	}
	return persons1_persons2_matchvalues;
}


HashMap<String, Double>  loadPredictedValues(Database db){
	
	HashMap<String, Double> persons1_persons2_predvalues = new HashMap();
	println "About to store the predicted values to the hashmap"
	for (GroundAtom atom : Queries.getAllAtoms(db, samePerson)){
		person1 = atom.arguments[0].toString().replace("'", "")
		person2 = atom.arguments[1].toString().replace("'", "")
		Double match_predicted_value = atom.getValue().toDouble()
		persons1_persons2_predvalues.put(person1+","+person2, match_predicted_value);
	}
	return persons1_persons2_predvalues;
}


//this function finds the best threshold value and returns it
double findThreshold(HashMap<String, Double> persons1_persons2_matchvalues, 
					 HashMap<String, Double> persons1_persons2_predvalues, 
					 Matches matches, double thresholdStep){
	println "About to find the best threshold value..."
	//find the best threshold value
	//run for all values from 0.01 with step 0.01
	double bestThreshold = 0.0;
	EvalResultsSingle bestResults = new EvalResultsSingle();
	ConfusionMatrix confMat = new ConfusionMatrix();
	for(double threshold=thresholdStep; threshold<1.0; threshold+=thresholdStep){
		println "Check for threshold value = " + threshold
		//Compute precision/recall/f-measure
		confMat = computeConfusionMatrix(persons1_persons2_predvalues, persons1_persons2_matchvalues, threshold, null);
		//println "Done with computing the confusion matrix -- now compute the stats for fold " + fold
		//compute stats call the function
		boolean update = computeStatsOneFold(confMat, matches, bestResults);
		if(update){
			bestThreshold = threshold;
			println "New threshold set at = " + bestThreshold;
		}
	}
	println "The best value for the threshold is " +  bestThreshold;
	return bestThreshold;
}

//Compute confusion matrix
//AND store the results to files
ConfusionMatrix computeConfusionMatrix(HashMap<String, Double> persons1_persons2_predvalues,
		HashMap<String, Double> persons1_persons2_matchvalues,
		double threshold, String fileName){
		
		ConfusionMatrix confMat = new ConfusionMatrix();
		
		//file where we will store the results
		def resultsFile;
		if(fileName!=null){
			resultsFile = new File(fileName);
			if(!resultsFile.createNewFile()){	//if the file already exists then delete it and create it
				resultsFile.delete();
				resultsFile.createNewFile();
			}
		}
		
		println "Inference results with learnt weights:"
		double match_true_value, match_predicted_value;
		int n=0;
		Set<String> persons = persons1_persons2_predvalues.keySet();
		for (String person1_person2 : persons){
			match_predicted_value = persons1_persons2_predvalues.get(person1_person2);
			//println person1 + "\t" + person2 + "\t>>\t"  + match_predicted_value;
			
			//search in the structure persons1_persons2_matchvalues for the pair <person1,person2>
			//and if it does exist then update the stats
			if(persons1_persons2_matchvalues.containsKey(person1_person2)){
				//get the value
				match_true_value = persons1_persons2_matchvalues.get(person1_person2);
				//now write the pair of persons in the file along with the value 1 if
				//this is a match and 0 otherwise
				//if(match_predicted_value>=threshold && fileName!=null){
				if(fileName!=null){
					String person1 = person1_person2.split(",")[0];
					String person2 = person1_person2.split(",")[1];
					resultsFile.append(person1 + "\t" + person2 + "\t" + match_predicted_value + "\n");
					//println "( " + person1_person2 + " ) = " + match_predicted_value + "\t" + match_true_value;
				}
				
				if(match_predicted_value>=threshold && match_true_value==1)
					confMat.classYes_predictedYes++;
				else if(match_predicted_value>=threshold && match_true_value==0)
					confMat.classNo_predictedYes++;
				else if(match_predicted_value<threshold && match_true_value==1)
					confMat.classYes_predictedNo++;
				else if(match_predicted_value<threshold && match_true_value==0)
					confMat.classNo_predictedNo++;
				else
					println "[ERROR]: In evaluation of the PSL model match_predicted_value="; //+ match_predicted_value
					//+ " match_true_value=" + match_true_value;
				n++;
			}
		}
		return confMat;
		
	}

	
	//function that computes stats
	boolean computeStatsOneFold(ConfusionMatrix confMat, Matches matches, 
							 EvalResultsSingle bestResults){
		
		if(confMat.classYes_predictedYes==0 ||confMat.classNo_predictedYes==0 ||
			confMat.classNo_predictedNo==0 || confMat.classNo_predictedYes==0)
			return false;
			
		//now compute the stats: precision, recall, f-measure
		double precisionYes = 1.0*confMat.classYes_predictedYes / (confMat.classYes_predictedYes + confMat.classNo_predictedYes);
		double precisionNo  = 1.0*confMat.classNo_predictedNo / (confMat.classNo_predictedNo + confMat.classYes_predictedNo);
		double recallYes    = 1.0*confMat.classYes_predictedYes / (confMat.classYes_predictedYes + confMat.classYes_predictedNo);
		double recallNo     = 1.0*confMat.classNo_predictedNo / (confMat.classNo_predictedYes + confMat.classNo_predictedNo);
			
		double fmeasureYes, fmeasureNo;
		if(precisionYes==0 && recallYes==0)
			fmeasureYes = 0;
		else
		   fmeasureYes = 2.0*precisionYes*recallYes / (precisionYes + recallYes);
		
		if(precisionNo==0 && recallNo==0)
		   fmeasureNo = 0;
		else
		   fmeasureNo  = 2.0*precisionNo*recallNo / (precisionNo + recallNo);
		
		double precision    = (matches.numberOfTrueMatches*precisionYes + matches.numberOfTrueNonMatches*precisionNo)/(matches.numberOfTrueMatches + matches.numberOfTrueNonMatches);
		double recall       = (matches.numberOfTrueMatches*recallYes + matches.numberOfTrueNonMatches*recallNo)/(matches.numberOfTrueMatches + matches.numberOfTrueNonMatches);
		double fmeasure     = (matches.numberOfTrueMatches*fmeasureYes + matches.numberOfTrueNonMatches*fmeasureNo)/(matches.numberOfTrueMatches + matches.numberOfTrueNonMatches);
		
		System.out.println("[DEBUG]: computePerformanceStatistics: Number of True Matches = " + matches.numberOfTrueMatches);
		System.out.println("[DEBUG]: computePerformanceStatistics: Number of True Non Matches = " + matches.numberOfTrueNonMatches);
		System.out.println("[DEBUG]: computePerformanceStatistics: classYes_predictedYes = " + confMat.classYes_predictedYes
				+ "\n\t\t\t classNo_predictedYes = " + confMat.classNo_predictedYes
				+ "\n\t\t\t classYes_predictedNo = " + confMat.classYes_predictedNo
				+ "\n\t\t\t classNo_predictedNo = " + confMat.classNo_predictedNo);
			
		System.out.format("\t\t Precision_Yes = %.3f\tPrecision_No = %.3f\n" +
				"\t\t Recall_Yes    = %.3f\tRecall_No    = %.3f\n" +
				"\t\t FMeasure_Yes  = %.3f\tFMeasure_No  = %.3f\n" +
				"\t\t Wght_Precision= %.3f\n" +
				"\t\t Wght_Recall   = %.3f\n" +
				"\t\t Wght_FMeasure = %.3f\n",
		  precisionYes, precisionNo,
		  recallYes, recallNo,
		  fmeasureYes, fmeasureNo,
		  precision,recall,fmeasure);
	  
		  //if this is true then update the class bestResults and return true
		  if(fmeasureYes>bestResults.fmeasure ||
			 (fmeasureYes == bestResults.fmeasure && recallYes > bestResults.recall) ||
			 (fmeasureYes == bestResults.fmeasure && recallYes==bestResults.recall) && precisionYes > bestResults.precision){
			 bestResults.setEvalResultsSingle(precisionYes, recallYes, fmeasureYes);
			 return true;
		  }
		  else
			  return false;
		
	}
	
	public void updateStatsAllFolds(EvalResults results, ConfusionMatrix confMat, 
								Matches matches){
		
		//now update the stats to compute the final results for precision, recall, f-measure
		
		//position 0 keeps the class NO MATCH
		//position 1 keeps the class MATCH
		//position 2 keeps the class NO MATCH
		
		double recallNo     = 1.0*confMat.classNo_predictedNo / (confMat.classNo_predictedYes + confMat.classNo_predictedNo);
		results.recall[0] += recallNo;
		results.squareRecall[0] += recallNo*recallNo;
		double recallYes    = 1.0*confMat.classYes_predictedYes / (confMat.classYes_predictedYes + confMat.classYes_predictedNo);
		results.recall[1] += recallYes;
		results.squareRecall[1] += recallYes*recallYes;
		double recallWeight       = (matches.numberOfTrueMatches*recallYes + matches.numberOfTrueNonMatches*recallNo)/(matches.numberOfTrueMatches + matches.numberOfTrueNonMatches);
		results.recall[2] += recallWeight;
		results.squareRecall[2] += recallWeight*recallWeight;
		
		double precisionNo  = 1.0*confMat.classNo_predictedNo / (confMat.classNo_predictedNo + confMat.classYes_predictedNo);
		results.precision[0]+= precisionNo;
		results.squarePrecision[0] += precisionNo*precisionNo;
		double precisionYes = 1.0*confMat.classYes_predictedYes / (confMat.classYes_predictedYes + confMat.classNo_predictedYes);
		results.precision[1]+=precisionYes;
		results.squarePrecision[1] += precisionYes*precisionYes;
		double precisionWeigtht    = (matches.numberOfTrueMatches*precisionYes + matches.numberOfTrueNonMatches*precisionNo)/(matches.numberOfTrueMatches + matches.numberOfTrueNonMatches);
		results.precision[2]+=precisionWeigtht;
		results.squarePrecision[2] += precisionWeigtht*precisionWeigtht;
		
		double fmeasureNo   = 2.0*precisionNo*recallNo / (precisionNo + recallNo);
		results.fmeasure[0] += fmeasureNo;
		results.squareFmeasure[0] += fmeasureNo*fmeasureNo;
		double fmeasureYes  = 2.0*precisionYes*recallYes / (precisionYes + recallYes);
		results.fmeasure[1] += fmeasureYes;
		results.squareFmeasure[1] += fmeasureYes*fmeasureYes;
		double fmeasureWeight     = (matches.numberOfTrueMatches*fmeasureYes + matches.numberOfTrueNonMatches*fmeasureNo)/(matches.numberOfTrueMatches + matches.numberOfTrueNonMatches);
		results.fmeasure[2] += fmeasureWeight;
		results.squareFmeasure[2] += fmeasureWeight*fmeasureWeight;
		
		
	}
	
	
	
		
	



