 /**
 * Copyright (c) DTAI - KU Leuven – All rights reserved.
 * Proprietary, do not copy or distribute without permission.
 * Written by Jessa Bekker and Pieter Robberechts
 */

import java.io.IOException;
import java.util.BitSet;

/**
 * This class is a bit slice for querying numerical attributes efficiently.
 */
public class BitSlice {

    private BitSet[] slice; // The bitslice, there is a slice for every bit of the attribute
    private int amplification; // store 2.70 as 270 => the amplification is 100

    /**
     * Constructor of the bit slice.
     *
     * @param data The original table that contains the attribute
     * @param attributeIndex The column number of the attribute in the table
     * @param encodingLength The number of bits to be used to encode the numbers
     * @param amplification The amplification of the values (used to encode doubles as integers)
     */
    public BitSlice(DataTable data, int attributeIndex, int encodingLength, int amplification) {
        this.amplification = amplification;
        this.slice = new BitSet[encodingLength];
        String [][] rawData = data.getData();

        for (int i = 0; i < encodingLength; i++) {
            slice[i] = new BitSet(data.size());
        }

        for (int i = 0; i < data.size(); i++) {
            double value = Double.parseDouble(rawData[i][attributeIndex]) * amplification;

            // To avoid floating point rounding errors, a small value should be added.
            // However, since the return type of query 3 is an int, cents are dropped anyway.
            //int intValue = (int) (value + 0.0000001); 
            int intValue = (int) value;
            
            for (int bit = 0; bit < encodingLength; bit++) {
                if ((intValue & (1 << bit)) != 0) {
                    slice[bit].set(i);
                }
            }
        }
    }


    /**
     * This returns the bit slice. Be careful, bit sets are mutable!
     * @return the bit slice
     */
    public BitSet[] getSlice() {
        return slice;
    }

    /**
     *This returns a subset of the bit slice. This is equivalent to setting the other values to zero.
     *
     * @param filter the bitset to use as filter. If a record is not in the filter, set the value to zero.
     * @return
     */
    // TODO make it static
    public BitSet[] getSubSlice(BitSet filter) {
        BitSet[] subset = new BitSet[slice.length];

        for (int i = 0; i < slice.length; i++) {
            subset[i] = (BitSet) slice[i].clone();
            subset[i].and(filter);
        }

        return  subset;
    }

    /**
     * This returns the amplification
     * @return amplification
     */
    public int getAmplification() {
        return amplification;
    }

    /**
     * This method answers the range query.
     * It returns the bitsets that respectively tell in which records the value is less than (LT), greater than (GT) and equal (EQ) to the number c.
     *
     * @param c the number to compare with
     * @param bitSlice the bit slice used to calculate the ranges in
     * @param amplification the amplification used on the numbers in the bit slice
     * @param nbRecords The number of records in the bitslice.
     * @return the selected records for [LT,GT,EQ]
     */
    
    public static BitSet[] range(double c, BitSet[] bitSlice, int amplification, int nbRecords) {
        BitSet lt = new BitSet(nbRecords);
        BitSet gt = new BitSet(nbRecords);
        BitSet eq = new BitSet(nbRecords);
        eq.set(0, nbRecords); 

        int intValue = (int) (c * amplification);
        int numBitSlices = bitSlice.length;

        // From lecture slides
        for (int i = numBitSlices - 1; i >= 0; i--) {
            BitSet Bi = bitSlice[i];
            boolean isBitSet = (intValue & (1 << i)) != 0;
    
            if (isBitSet) {
                // lt = lt OR (eq AND NOT(Bi))
                BitSet temp = (BitSet) eq.clone();
                temp.andNot(Bi);  
                lt.or(temp);  
        
                // eq = eq AND Bi
                eq.and(Bi);    
        
            } else {
                // gt = gt OR (eq AND Bi)
                BitSet temp = (BitSet) eq.clone();
                temp.and(Bi);
                gt.or(temp);
        
                // eq = eq AND NOT(Bi)
                eq.andNot(Bi);
            }
        }
    
        return new BitSet[]{lt, gt, eq};
    }


    /**
     * This method calculates the sum of the numbers in the bit slice.
     *
     * @param bitSlice the bitslice
     * @return the sum of the numbers in the bit slice
     */
    public static int sum(BitSet[] bitSlice) {
        int sum = 0;
        int encodingLength = bitSlice.length;

        for (int bit = 0; bit < encodingLength; bit++) {
            sum += bitSlice[bit].cardinality() * (1 << bit);
        }

        return sum;
    }
}
