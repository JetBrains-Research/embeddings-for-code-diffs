
        public boolean isFeasible(final double[] x) {
            if (boundaries == null) {
                return true;
            }

            final double[] bLoEnc = encode(boundaries[0]);
            final double[] bHiEnc = encode(boundaries[1]);

            for (int i = 0; i < x.length; i++) {
                if (x[i] < bLoEnc[i]) {
                    return false;
                }
                if (x[i] > bHiEnc[i]) {
                    return false;
                }
            }
            return true;
        }