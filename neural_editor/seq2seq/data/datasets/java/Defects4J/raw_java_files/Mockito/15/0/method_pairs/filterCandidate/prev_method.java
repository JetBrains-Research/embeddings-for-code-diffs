public OngoingInjecter filterCandidate(final Collection<Object> mocks, final Field field, final Object fieldInstance) {
        if(mocks.size() == 1) {
            final Object matchingMock = mocks.iterator().next();

            return new OngoingInjecter() {
                public boolean thenInject() {
                    try {
                            new FieldSetter(fieldInstance, field).set(matchingMock);
                    } catch (Exception e) {
                        throw new MockitoException("Problems injecting dependency in " + field.getName(), e);
                    }
                    return true;
                }
            };
        }

        return new OngoingInjecter() {
            public boolean thenInject() {
                return false;
            }
        };

    }