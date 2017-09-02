package edu.ucsc.NIH;

public class EvalResults {

	Double[] precision = [0.0,0.0,0.0]; //first is for the class no match, second for the class  match, third is the weighted avg
    Double[]  recall= [0.0,0.0,0.0];//same
    Double[]  fmeasure= [0.0,0.0,0.0];//same
    
    //these are for the calculation of the standard deviation
    Double[] squarePrecision= [0.0,0.0,0.0]; //first is for the class no match, second for the class  match, third is the weighted avg
    Double[] squareRecall= [0.0,0.0,0.0];//same
    Double[]  squareFmeasure= [0.0,0.0,0.0];//same
    
    //standard deviation
    Double[]  sdPrecision= [0.0,0.0,0.0]; //first is for the class no match, second for the class match, third is the weighted avg
    Double[]  sdRecall= [0.0,0.0,0.0];//same
    Double[]  sdFmeasure= [0.0,0.0,0.0];//same
    
    /*public evalResults(){
        precision = new double[3]; //first is for the class no match, second for the class  match, third is the weighted avg
        recall = new double[3];//same
        fmeasure = new double[3];//same
        
        sdPrecision = new double[3]; //first is for the class no match, second for the class  match, third is the weighted avg
        sdRecall = new double[3];//same
        sdFmeasure = new double[3];//same
        
        squarePrecision = new double[3]; //first is for the class match, second for the class no match, third is the weighted avg
        squareRecall = new double[3];//same
        squareFmeasure = new double[3];//same
       
        //initialize everything to zero
        for(int i=0;i<3;i++){
            precision[i] = 0.0; //first is for the class NO match, second for the class match, third is the weighted avg
            recall[i] = 0.0;//same
            fmeasure[i] = 0.0;//same
        
            sdPrecision[i] = 0.0; //first is for the class match, second for the class no match, third is the weighted avg
            sdRecall[i] = 0.0;//same
            sdFmeasure[i] = 0.0;//same
            
            squarePrecision[i] = 0.0; //first is for the class match, second for the class no match, third is the weighted avg
            squareRecall[i] = 0.0;//same
            squareFmeasure[i] = 0.0;//same
            
        }
    }*/
    
    public void computeStandardDeviation(){
        for(int i=0;i<3;i++){
            sdPrecision[i] = Math.sqrt((1/5.0)*squarePrecision[i]-Math.pow(precision[i]/5.0,2));
            sdRecall[i] = Math.sqrt((1/5.0)*squareRecall[i]-Math.pow(recall[i]/5.0,2));
            sdFmeasure[i] = Math.sqrt((1/5.0)*squareFmeasure[i]-Math.pow(fmeasure[i]/5.0,2));
            
        }
    }
	
	public void printResults3Digits(){
		
		//compute the SD
		this.computeStandardDeviation();
		System.out.println("\n\n----FINAL RESULTS----\n\n")
		System.out.println(String.format("Precision NO MATCH = %.3f" +
				" (SD) = %.3f" +
				"\tRecall NO MATCH = %.3f"   +
				" (SD) = %.3f" +
				"\tFMeasure NO MATCH = %.3f"+
				" (SD) = %.3f",
				this.precision[0]/5.0,
				this.sdPrecision[0],
				this.recall[0]/5.0,
				this.sdRecall[0],
				this.fmeasure[0]/5.0,
				this.sdFmeasure[0]
				));
		
		System.out.println(String.format("Precision MATCH = %.3f" +
				" (SD) = %.3f" +
				"\tRecall MATCH = %.3f"   +
				" (SD) = %.3f" +
				"\tFMeasure MATCH = %.3f"+
				" (SD) = %.3f",
				this.precision[1]/5.0,
				this.sdPrecision[1],
				this.recall[1]/5.0,
				this.sdRecall[1],
				this.fmeasure[1]/5.0,
				this.sdFmeasure[1]
				));
		
		System.out.println(String.format("Precision AVG = %.3f" +
				" (SD) = %.3f" +
				"\tRecall AVG = %.3f"   +
				" (SD) = %.3f" +
				"\tFMeasure AVG = %.3f"+
				" (SD) = %.3f",
				this.precision[2]/5.0,
				this.sdPrecision[2],
				this.recall[2]/5.0,
				this.sdRecall[2],
				this.fmeasure[2]/5.0,
				this.sdFmeasure[2]
				));
		
	}
	
	public class EvalResultsSingle{
		Double precision;
		Double recall;
		Double fmeasure;
		
		public EvalResultsSingle(){
			this.precision = 0.0;
			this.recall = 0.0;
			this.fmeasure = 0.0;
		}
		
		public setEvalResultsSingle(double prec, double rec, double fm){
			this.precision = prec;
			this.recall = rec;
			this.fmeasure = fm;
		}
	}
	
	public class Matches{
		int numberOfTrueMatches;
		int numberOfTrueNonMatches;
		
		public Matches(){
			this.numberOfTrueMatches = 0;
			this.numberOfTrueNonMatches = 0;
		}
		
		public setMatches(int trueMatches, int trueNonMatches){
			this.numberOfTrueMatches = trueMatches;
			this.numberOfTrueNonMatches = trueNonMatches;
		}
	}
	
}
