public boolean thenInject() {
                    try {
                        if (!new BeanPropertySetter(fieldInstance, field).set(matchingMock)) {
                            new FieldSetter(fieldInstance, field).set(matchingMock);
                        }
                    } catch (Exception e) {
                        throw new MockitoException("Problems injecting dependency in " + field.getName(), e);
                    }
                    return true;
                }