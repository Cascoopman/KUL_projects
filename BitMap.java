import java.io.IOException;
import java.util.*;

/**
 * This class is a bitmap for querying the bitMap efficiently.
 *
 * @author Jessa Bekker.
 */
public class BitMap {

    private BitSet[] bitMap; // The bitmap, there is a bit set for every possible value of the attribute
    private Map<String, Integer> value2position; // This maps the possible values for the attribute to their position in the bitmap
    
    /**
     * Constructor of the bit map
     *
     * @param data The original table that contains the attribute
     * @param attributeIndex The column number of the attribute in the table
     * @throws IOException
     */
    public BitMap(DataTable data, int attributeIndex) throws IOException {
        value2position = new HashMap<>();
        int numRecords = data.size();
        String[][] rawData = data.getData();
        int position = 0;
        bitMap = new BitSet[0];

        // Process each record, assigning unique positions and setting bits
        for (int i = 0; i < numRecords; i++) {
            String value = rawData[i][attributeIndex];
            Integer pos = value2position.get(value);

            if (pos == null) {
                pos = position++;
                value2position.put(value, pos);
                bitMap = Arrays.copyOf(bitMap, position);
                bitMap[pos] = new BitSet(numRecords); 
            }
            bitMap[pos].set(i);
        }
    }

    /**
     * This returns the bit map. Be careful, bit sets are mutable!
     * @return the bit map
     */
    public BitSet[] getBitMap() {
        return bitMap;
    }

    /**
     * This returns the position of the given value of the attribute that is represented by this bitmap
     * @param value the value
     * @return the position of the given value
     */
    public int getPositionOf(String value) {
        return value2position.get(value);
    }


    /**
     * This method returns the records where the attribute of this bit map takes the ith value.
     *
     * @param bitmap the bitmap to select from
     * @param i the value to select on
     * @return the selected records
     */
    public static BitSet select(BitSet[] bitmap, int i) {
        return bitmap[i];
    }


    /**
     * This method returns the number of records  where the attribute of this bit map takes the ith value.
     *
     * @param bitmap the bitmap to count in
     * @param i the value to select on
     * @return the number of records that have the given attribute value
     */
    public static int count(BitSet[] bitmap, int i) {
        return bitmap[i].cardinality();
    }
}
