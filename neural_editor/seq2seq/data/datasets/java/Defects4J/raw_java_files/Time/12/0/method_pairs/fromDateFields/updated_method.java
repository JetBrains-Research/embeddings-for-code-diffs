
    @SuppressWarnings("deprecation")
    public static LocalDate fromDateFields(Date date) {
        if (date == null) {
            throw new IllegalArgumentException("The date must not be null");
        }
        if (date.getTime() < 0) {
            // handle years in era BC
            GregorianCalendar cal = new GregorianCalendar();
            cal.setTime(date);
            return fromCalendarFields(cal);
        }
        return new LocalDate(
            date.getYear() + 1900,
            date.getMonth() + 1,
            date.getDate()
        );
    }