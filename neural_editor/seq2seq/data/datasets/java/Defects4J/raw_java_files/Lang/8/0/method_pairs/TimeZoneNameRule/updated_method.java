
        TimeZoneNameRule(TimeZone timeZone, Locale locale, int style) {
            mLocale = locale;
            mStyle = style;
            
            mStandard = getTimeZoneDisplay(timeZone, false, style, locale);
            mDaylight = getTimeZoneDisplay(timeZone, true, style, locale);
        }