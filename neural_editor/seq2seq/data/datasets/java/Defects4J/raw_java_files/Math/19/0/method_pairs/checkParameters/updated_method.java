
    private void checkParameters() {
        final double[] init = getStartPoint();
        final double[] lB = getLowerBound();
        final double[] uB = getUpperBound();

        // Checks whether there is at least one finite bound value.
        boolean hasFiniteBounds = false;
        for (int i = 0; i < lB.length; i++) {
            if (!Double.isInfinite(lB[i]) ||
                !Double.isInfinite(uB[i])) {
                hasFiniteBounds = true;
                break;
            }
        }
        // Checks whether there is at least one infinite bound value.
        boolean hasInfiniteBounds = false;
        if (hasFiniteBounds) {
            for (int i = 0; i < lB.length; i++) {
                if (Double.isInfinite(lB[i]) ||
                    Double.isInfinite(uB[i])) {
                    hasInfiniteBounds = true;
                    break;
                }
            }

            if (hasInfiniteBounds) {
                // If there is at least one finite bound, none can be infinite,
                // because mixed cases are not supported by the current code.
                throw new MathUnsupportedOperationException();
            } else {
                // Convert API to internal handling of boundaries.
                boundaries = new double[2][];
                boundaries[0] = lB;
                boundaries[1] = uB;

                // Abort early if the normalization will overflow (cf. "encode" method).
                for (int i = 0; i < lB.length; i++) {
                    if (Double.isInfinite(boundaries[1][i] - boundaries[0][i])) {
                        final double max = Double.MAX_VALUE + boundaries[0][i];
                        final NumberIsTooLargeException e
                            = new NumberIsTooLargeException(boundaries[1][i],
                                                            max,
                                                            true);
                        e.getContext().addMessage(LocalizedFormats.OVERFLOW);
                        e.getContext().addMessage(LocalizedFormats.INDEX, i);

                        throw e;
                    }
                }
            }
        } else {
            // Convert API to internal handling of boundaries.
            boundaries = null;
        }

        if (inputSigma != null) {
            if (inputSigma.length != init.length) {
                throw new DimensionMismatchException(inputSigma.length, init.length);
            }
            for (int i = 0; i < init.length; i++) {
                if (inputSigma[i] < 0) {
                    throw new NotPositiveException(inputSigma[i]);
                }
                if (boundaries != null) {
                    if (inputSigma[i] > boundaries[1][i] - boundaries[0][i]) {
                        throw new OutOfRangeException(inputSigma[i], 0, boundaries[1][i] - boundaries[0][i]);
                    }
                }
            }
        }
    }