
    public static double binomialCoefficientDouble(final int n, final int k) {
        
  
        return Math.floor(Math.exp(binomialCoefficientLog(n, k)) + 0.5);
    }