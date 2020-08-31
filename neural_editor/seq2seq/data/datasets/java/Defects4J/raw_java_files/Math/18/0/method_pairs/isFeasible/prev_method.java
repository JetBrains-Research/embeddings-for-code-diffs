
        public boolean isFeasible(final double[] x) {
            if (boundaries == null) {
                return true;
            }


            for (int i = 0; i < x.length; i++) {
                if (x[i] < 0) {
                    return false;
                }
                if (x[i] > 1.0) {
                    return false;
                }
            }
            return true;
        }