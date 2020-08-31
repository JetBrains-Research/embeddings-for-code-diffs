
    public static double binomialCoefficientLog(final int n, final int k) {
        if (n < k) {
            throw new IllegalArgumentException(
                "must have n >= k for binomial coefficient (n,k)");
        }
        if (n < 0) {
            throw new IllegalArgumentException(
                "must have n >= 0 for binomial coefficient (n,k)");
        }
        if ((n == k) || (k == 0)) {
            return 0;
        }
        if ((k == 1) || (k == n - 1)) {
            return Math.log((double) n);
        }
        
        /*
         * For values small enough to do exact integer computation,
         * return the log of the exact value 
         */
        
        /*
         * Return the log of binomialCoefficientDouble for values that will not
         * overflow binomialCoefficientDouble
         */
        
        /*
         * Sum logs for values that could overflow
         */
        double logSum = 0;

        // n!/k!
        for (int i = k + 1; i <= n; i++) {
            logSum += Math.log((double)i);
        }

        // divide by (n-k)!
        for (int i = 2; i <= n - k; i++) {
            logSum -= Math.log((double)i);
        }

        return logSum;      
    }