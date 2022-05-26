/*
The copyrights of this software are owned by Duke University.
Please refer to the LICENSE and README.md files for licensing instructions.
The source code can be found on the following GitHub repository: https://github.com/wmglab-duke/ascent
*/

package model;

import com.comsol.model.*;
import com.comsol.model.physics.PhysicsFeature;
import com.comsol.model.util.ModelUtil;
import com.comsol.util.exceptions.FlException;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.text.DecimalFormat;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.regex.Pattern;

/**
 * model.ModelWrapper
 *
 * Master high-level class for managing a model, its metadata, and various critical operations such as creating parts,
 * assigning physics, and extracting potentials. This class houses the "meaty" operations of actually interacting with
 * the model object when creating parts in the static class model.Parts.
 */

@SuppressWarnings({"unchecked","rawtypes","path"})
public class ModelWrapper {

    // UNION PSEUDONYM CONSTANTS
    public static final String ALL_NERVE_PARTS_UNION = "allNervePartsUnion";
    public static final String ENDO_UNION = "endoUnion";
    public static final String PERI_UNION = "periUnion";

    // if these change, also need to change in createEnvironmentPartInstance
    public static final String DISTAL_MEDIUM = "DistalMedium";
    public static final String PROXIMAL_MEDIUM = "ProximalMedium";

    public static final String[] ALL_UNIONS = new String[]{
            ModelWrapper.ENDO_UNION,
            ModelWrapper.ALL_NERVE_PARTS_UNION,
            ModelWrapper.PERI_UNION
    };
    // associated union contributors for above constants
    private HashMap<String, ArrayList<String>> unionContributors = new HashMap<>();

    // INSTANCE VARIABLES
    private Model model; // model

    public IdentifierManager im = new IdentifierManager(); // top level identifier manager
    public static IdentifierManager ve_im = new IdentifierManager(); // top level identifier manager

    private HashMap<String, IdentifierManager> partPrimitiveIMs = new HashMap<>(); // for managing parts within COMSOL

    // directory structure
    private String root;
    private String dest;

    // CONSTRUCTORS
    /**
     * Default constructor (minimum of 2 arguments)
     * @param model com.comsol.model.Model object is REQUIRED
     * @param projectRoot the root directory of the project (might remove if unnecessary)
     */
    ModelWrapper(Model model, String projectRoot) {
        this.model = model;
        this.root = projectRoot;
        this.initUnionContributors();
    }

    /**
     * Overloaded constructor for passing in save directory
     * @param model com.comsol.model.Model object is REQUIRED
     * @param projectRoot the root directory of the project (might remove if unnecessary)
     * @param defaultSaveDestination directory in which to save (relative to project root)
     */
    ModelWrapper(Model model, String projectRoot, String defaultSaveDestination) {
        this(model, projectRoot);
        this.initUnionContributors();
        this.dest = defaultSaveDestination;
    }

    // ACCESSOR/MUTATOR METHODS
    /**
     * @return the model
     */
    public Model getModel() {
        return model;
    }

    /**
     * @return the root of the project (String path)
     */
    public String getRoot() {
        return root;
    }

    /**
     * @return the destination path to which to save the model
     */
    public String getDest() {
        return dest;
    }

    /**
     * @param root set the project root (String path)
     */
    public void setRoot(String root) {
        this.root = root;
    }

    /**
     * @param dest set the destination path to which to save the model
     */
    public void setDest(String dest) {
        this.dest = dest;
    }

    // OTHER METHODS
    /**
     * call method on im (IdentifierManager)... see class for details
     */
    public String next(String key) {
        return this.im.next(key);
    }

    /**
     * call method on im (IdentifierManager)... see class for details
     */
    public String next(String key, String pseudonym) {
        return this.im.next(key, pseudonym);
    }

    /**
     * call method on im (IdentifierManager)... see class for details
     */
    public String get(String psuedonym) {
        return this.im.get(psuedonym);
    }

    /**
     * @param partPrimitiveLabel the name of the part primitive (i.e. "TubeCuff_Primitive")
     * @return the associated IdentifierManager, for correct intra-part indexing
     */
    public IdentifierManager getPartPrimitiveIM(String partPrimitiveLabel) {
        return this.partPrimitiveIMs.get(partPrimitiveLabel);
    }

    /**
     * @param destination full path to save to
     * @return success indicator
     */
    public boolean save(String destination) {
        try {
            this.model.save(destination);
            return true;
        } catch (IOException e) {
            e.printStackTrace();
            return false;
        }
    }

    /**
     * Convenience method for saving to relative directory (this.dest) wrt the project directory (root)
     * @return success indicator
     */
    public boolean save() {
        if (this.dest != null) return save(String.join("/", new String[]{this.root, this.dest}));
        else {
            System.out.println("Save directory not initialized");
            return false;
        }
    }

    /**
     * Create the required primitives for a given cuff json
     * @param name json filename WITH extension (i.e. "LivaNova2000.json")
     * @return success indicator
     */
    public boolean addCuffPartPrimitives(String name) {
        // extract data from json
        try {
            JSONObject cuffData = JSONio.read(
                    String.join("/", new String[]{this.root, "config", "system", "cuffs", name})
            );

            // get the id for the next "par" (i.e., parameters section), and give it a name from the JSON file name
            String id = this.next("par", name);
            model.param().group().create(id);
            model.param(id).label(name.split("\\.")[0] + " Parameters");

            // loop through all parameters in file, and set in parameters
            for (Object item : (JSONArray) cuffData.get("params")) {
                JSONObject itemObject = (JSONObject) item;
                model.param(id).set(
                        (String) itemObject.get("name"),
                        (String) itemObject.get("expression"),
                        (String) itemObject.get("description")
                );
            }

            // for each required part primitive, create it (if not already existing)
            for (Object item: (JSONArray) cuffData.get("instances")) {
                JSONObject itemObject = (JSONObject) item;
                String partPrimitiveName = (String) itemObject.get("type"); // quick cast to String

                // create the part primitive if it has not already been created
                if (! this.im.hasPseudonym(partPrimitiveName)) {
                    // get next available (TOP LEVEL) "part" id
                    String partID = this.im.next("part", partPrimitiveName);
                    try {
                        // TRY to create the part primitive (catch error if no existing implementation)
                        IdentifierManager partPrimitiveIM = Part.createCuffPartPrimitive(partID, partPrimitiveName, this);

                        // add the returned id manager to the HashMap of IMs with the partName as its key
                        this.partPrimitiveIMs.put(partPrimitiveName, partPrimitiveIM);

                    } catch (IllegalArgumentException e) {
                        e.printStackTrace();
                        return false;

                    }
                }
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
            return false;
        }
        return true;
    }

    /**
     * Instantiate required primitives for given cuff
     * NOTE: addCuffPartPrimitives() MUST be called first or there will be no primitives to instantiate
     * @param name same formatting as in addCuffPartPrimitives()
     * @param modelData
     * @return success indicator
     */
    public boolean addCuffPartInstances(String name, JSONObject modelData) {
        // extract data from json (name is something like Enteromedics.json)
        try {
            JSONObject cuffData = JSONio.read(
                    String.join("/", new String[]{this.root, "config", "system", "cuffs", name})
            );

            // loop through all part instances
            for (Object item: (JSONArray) cuffData.get("instances")) {
                JSONObject itemObject = (JSONObject) item;

                String instanceLabel = (String) itemObject.get("label");
                String instanceID = this.im.next("pi", instanceLabel);
                String type = (String) itemObject.get("type");
                Part.createCuffPartInstance(instanceID, instanceLabel, type , this, itemObject);
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
            return false;
        }
        return true;
    }

    /**
     * Assign previously defined materials to domains in part instances
     * @param cuffData is loaded JSON data for a defined cuff
     */
    public boolean addCuffPartMaterialAssignments(JSONObject cuffData) {
        // extract data from json, its name is something like Enteromedics.json
        // loop through all part instances
        for (Object item: (JSONArray) cuffData.get("instances")) {
            JSONObject itemObject = (JSONObject) item;

            String instanceLabel = (String) itemObject.get("label");
            String type = (String) itemObject.get("type");
            Part.addCuffPartMaterialAssignment(instanceLabel, type, this, itemObject);
        }
        return true;
    }

    /**
     * Create materials necessary for fascicles, nerve, surrounding media, etc.
     * @return success indicator
     */
    public boolean addMaterialDefinitions(ArrayList<String> materials, JSONObject modelData, ModelParamGroup materialParams) {
        try {
            // load system defined materials JSON into memory
            JSONObject materialsData = JSONio.read(
                    String.join("/", new String[]{this.root, "config", "system", "materials.json"})
            );

            // add material definition for those materials that are needed in the instantiated parts
            for (String function:materials) {
                if (! this.im.hasPseudonym(function)) {
                    String materialID = this.im.next("mat", function);
                    Part.defineMaterial(materialID, function, modelData, materialsData, this, materialParams);
                }
            }

        } catch (FileNotFoundException e) {
            e.printStackTrace();
            return false;
        }
        return true;
    }

        /**
         * @return success indicator
         */
    public static double[] extractPotentials(Model model, String coords_path) throws IOException {

        // Load coordinates (x,y,z) from file in form: top line is number of rows of coords (int)
        //                                             coordinates[0][i] = [x] in micron, (double)
        //                                             coordinates[1][i] = [y] in micron, (double)
        //                                             coordinates[2][i] = [z] in micron  (double)

        // read in coords for axon segments as defined and saved to file in Python
        double[][] coordinatesLoaded;
        coordinatesLoaded = readCoords(coords_path);

        // transpose saved coordinates (we like to save (x,y,z) as column vectors, but COMSOL wants as rows)
        double[][] coordinates;
        coordinates = transposeMatrix(coordinatesLoaded);

        // get Ve from COMSOL

        String id = ve_im.next("interp");
        model.result().numerical().create(id, "Interp");
        model.result().numerical(id).set("expr", "V");
        model.result().numerical(id).set("recover", "pprint");
        model.result().numerical(id).set("matherr", "on");
        model.result().numerical(id).setInterpolationCoordinates(coordinates);
        double[][][] ve_pre = model.result().numerical(id).getData();
        int len = ve_pre[0][0].length; // number of coordinates

        double[] ve = new double[len];
        for (int i = 0; i < len; i++) {
            ve[i] = ve_pre[0][0][i];
        }

        return ve;
    }

    /**
     * For each fiber set created on the Python side of things, extract potentials and save to file
     * @param projectPath
     * @param run_path
     */
    public static void extractAllPotentials(String projectPath, String run_path, String modelStr) throws IOException {
        System.out.println("\tExtracting/writing all potentials - skips if file already exists");

        // READ IN RUN CONFIGURATION DATA
        JSONObject runData = JSONio.read(run_path);
        // get sample number
        int sample = runData.getInt("sample");
        // get sims list
        JSONArray sims_list = runData.getJSONArray("sims");

        // GET BASES FOR EACH MODEL (SET UP SO THAT LOADS EACH BASE MPH FILE ONLY ONCE => SPEED)
        // load model config data
        String model_path = String.join("/", new String[]{ // build path to sim config file
                projectPath, "samples", Integer.toString(sample), "models", modelStr
        });
        String model_config_path = String.join("/", new String[]{ // build path to sim config file
                model_path, "model.json"
        });
        JSONObject modelData = JSONio.read(model_config_path); // load sim configuration data

        // construct bases path (to MPH)
        String bases_directory = String.join("/", new String[]{
                model_path, "bases"
        });

        // get bases at the bases MPH path
        String[] bases_paths = new File(bases_directory).list();
        assert bases_paths != null;
        bases_paths = Arrays.stream(bases_paths).filter(s -> Pattern.matches("[0-9]+\\.mph", s)).toArray(String[]::new);

        double[][][][][] bases = new double[bases_paths.length][][][][];
        double[][][][][] ss_bases = new double[bases_paths.length][][][][];

        for (int basis_ind = 0; basis_ind < bases_paths.length; basis_ind++) { // loop over bases

            // LOAD BASIS MPH MODEL
            String basis_dir = String.join("/", new String[]{
                    bases_directory, bases_paths[basis_ind]
            });
            File file = new File(basis_dir);
            while(!file.canWrite() || !file.canRead()) {
                System.out.println("\twaiting");
            }
            String model_tag = ModelUtil.uniquetag("Model");
            Model basis = ModelUtil.load(model_tag, basis_dir);

            bases[basis_ind] = new double[sims_list.length()][][][];
            ss_bases[basis_ind] = new double[sims_list.length()][][][];

            double[][][][] sim = bases[basis_ind]; // pointer
            double[][][][] ss_sim = ss_bases[basis_ind]; // pointer

            for (int sim_ind = 0; sim_ind < sims_list.length(); sim_ind++) { // loop over sims
                int sim_num = (int) sims_list.get(sim_ind); // get sim number for index in sims list

                // build path to directory of sim
                String sim_dir = String.join("/", new String[]{
                        model_path, "sims", Integer.toString(sim_num)
                });

                // build path to directory of fibersets
                String coord_dir = String.join("/", new String[]{
                        sim_dir, "fibersets"
                });

                // build path to directory of fibersets
                String ss_coord_dir = String.join("/", new String[]{
                        sim_dir, "ss_coords"
                });

                // build path to directory of ve for each fiberset
                String ve_dir = String.join("/", new String[]{
                        sim_dir, "potentials"
                });

                // build path to directory of ve for each ss_fiberset
                String ss_ve_dir = String.join("/", new String[]{
                        sim_dir, "ss_bases"
                });

                // build path to key (fiberset x srcs) file
                String key_path = String.join("/", new String[]{
                        ve_dir, "key.dat"
                });

                // if sim potentials directory does not yet exist, make it
                File vePathFile = new File(ve_dir);
                if (!vePathFile.exists()) {
                    boolean success = vePathFile.mkdirs();
                    assert success;
                }

                // load key (fiberset x srcs) file
                File f_key = new File(key_path);
                Scanner scan_key = new Scanner(f_key);

                // save rows (number of coords) at top line... so number of lines in file is (number of coords +1)
                String products = scan_key.nextLine();
                int n_products = Integer.parseInt(products.trim());

                // pre-allocated array of doubles for products in file
                // (2 columns by default for (active_src_select,fiberset_select)
                int[][] prods = new int[n_products][2];
                int row_ind = 0;

                // assign contents of key (fiberset x srcs) file to array
                String thisLine;
                while (scan_key.hasNextLine()) { // while there are more lines to scan
                    thisLine = scan_key.nextLine();
                    String[] parts = thisLine.split("\\s+");
                    for (int i = 0; i < parts.length; i++) {
                        prods[row_ind][i] = Integer.parseInt(parts[i]);
                    }
                    row_ind++;
                }

                // find the max fibersets index (max in 2nd column) from prods (loaded from key.dat)
                int n_fibersets = prods[0][1];
                for(int i = 0 ; i < prods.length ; i++) {
                    if (prods[i][1] > n_fibersets){
                        n_fibersets = prods[i][1];
                    }
                }
                n_fibersets++;

                File ss_coords_dir = new File(ss_coord_dir);
                int n_ss_fibersets;
                if (ss_coords_dir.exists()) {
                    File[] ss_coords_dir_List = ss_coords_dir.listFiles();
                    n_ss_fibersets = ss_coords_dir_List.length;

                    ss_sim[sim_ind] = new double[n_ss_fibersets][][];
                    double[][][] ss_fiberset = ss_sim[sim_ind]; // pointer

                    // SUPER SAMPLING
                    // create list of fiber coords (one for each fiber)

                    File ss_f_coords = new File(ss_coord_dir);
                    String[] ss_fiber_coords_list = ss_f_coords.list();

                    assert ss_fiber_coords_list != null;

                    ss_fiberset[0] = new double[ss_fiber_coords_list.length][];
                    double[][] ss_fibers = ss_fiberset[0]; // pointer

                    for (int ss_fiber_ind = 0; ss_fiber_ind < ss_fiber_coords_list.length; ss_fiber_ind++) { // loop over fiber coords in list of fiber coords
                        String ss_fiber_coords = ss_fiber_coords_list[ss_fiber_ind];

                        String[] ss_fiber_file_parts = ss_fiber_coords.split("\\.");
                        Integer ss_fiber_file_ind = Integer.parseInt(ss_fiber_file_parts[0]);

                        String ss_coord_path = String.join("/", new String[]{
                                ss_coord_dir, ss_fiber_coords
                        }); // build path to coordinates

                        ss_fibers[ss_fiber_file_ind] = extractPotentials(basis, ss_coord_path);

                        // if ss_potentials directory does not yet exist, make it
                        File ss_vePathFile = new File(ss_ve_dir);
                        if (!ss_vePathFile.exists()) {
                            boolean success = ss_vePathFile.mkdirs();
                            assert success;
                        }

                        // build path to directory of fibersets
                        String ss_ve_fiberset_basis_dir = String.join("/", new String[]{
                                sim_dir,
                                "ss_bases",
                                Integer.toString(basis_ind)
                        });

                        // if ss_fiberset_basis_potentials directory does not yet exist, make it
                        File ss_ve_fiberset_basis_dirPathFile = new File(ss_ve_fiberset_basis_dir);
                        if (!ss_ve_fiberset_basis_dirPathFile.exists()) {
                            boolean success = ss_ve_fiberset_basis_dirPathFile.mkdirs();
                            assert success;
                        }

                        String ss_ve_path = String.join("/", new String[]{
                                ss_ve_fiberset_basis_dir, ss_fiber_file_ind + ".dat"
                        });

                        if (new File(ss_ve_path).exists()) {
                            continue;
                        }
                        writeVe(ss_fibers[ss_fiber_file_ind], ss_ve_path);

                    }
                }

                sim[sim_ind] = new double[n_fibersets][][];
                double[][][] fiberset = sim[sim_ind]; // pointer

                for (int fiberset_ind = 0; fiberset_ind < n_fibersets; fiberset_ind++) { // loop over fibersets

                    // create list of fiber coords (one for each fiber)
                    String fiberset_dir = String.join("/", new String[]{
                            coord_dir, Integer.toString(fiberset_ind)
                    });
                    File f_coords = new File(fiberset_dir);
                    String[] fiber_coords_list = f_coords.list();

                    assert fiber_coords_list != null;
                    fiber_coords_list = Arrays.stream(fiber_coords_list).filter(s -> Pattern.matches("[0-9]+\\.dat", s)).toArray(String[]::new);

                    assert fiber_coords_list != null;

                    fiberset[fiberset_ind] = new double[fiber_coords_list.length][];
                    double[][] fibers = fiberset[fiberset_ind]; // pointer

                    for (int fiber_ind = 0; fiber_ind < fiber_coords_list.length; fiber_ind++) { // loop over fiber coords in list of fiber coords

                        String fiber_coords = fiber_coords_list[fiber_ind];

                        String[] fiber_file_parts = fiber_coords.split("\\.");
                        Integer fiber_file_ind = Integer.parseInt(fiber_file_parts[0]);

                        String coord_path = String.join("/", new String[]{
                                fiberset_dir, fiber_coords
                        }); // build path to coordinates

                        fibers[fiber_file_ind] = extractPotentials(basis, coord_path);

                    }
                }
            }
            // remove basis from memory
            ModelUtil.remove(basis.tag());
        }

        String cuff = modelData.getJSONObject("cuff").getString("preset");

        // COMBINE AND MAKE POTENTIALS FOR N_SIMS FROM BASES
        double[][][][] final_ve = new double[sims_list.length()][][][];
        for (int basis_ind = 0; basis_ind < bases_paths.length; basis_ind++) { // loop over bases
            for (int sim_ind = 0; sim_ind < sims_list.length(); sim_ind++) { // loop over sims
                // get sim number for index in sims list and load sim configuration data
                int sim_num = (int) sims_list.get(sim_ind);
                // build path to sim config file, load sim configuration data
                String sim_config_path = String.join("/", new String[]{
                        projectPath, "config", "user", "sims", sim_num + ".json"
                });
                JSONObject simData = JSONio.read(sim_config_path);

                // get array of contact combo weightings
                JSONObject active_srcs = simData.getJSONObject("active_srcs");

                // if the active_srcs weightings have been assigned, use the ones that match the cuff,
                // otherwise, attempt to use "default"
                JSONArray src_combo_list;
                if (active_srcs.has(cuff)) {
                    src_combo_list = active_srcs.getJSONArray(cuff);
                    System.out.println("\tFound the assigned contact weighting for " + cuff + " in sim " + sim_num + " config file");
                } else {
                    src_combo_list = active_srcs.getJSONArray("default");
                    System.out.println("\tWARNING: did NOT find the assigned contact weighting for " + cuff +
                            " in sim " + sim_num + " config file, moving forward with DEFAULT (use with caution)");
                }

                // build path to directory of sim
                String sim_dir = String.join("/", new String[]{
                        projectPath, "samples", Integer.toString(sample), "models", modelStr,
                        "sims", Integer.toString(sim_num)
                });

                // build path to directory of fibersets
                String coord_dir = String.join("/", new String[]{
                        sim_dir, "fibersets"
                });

                // build path to directory of key (fiberset x srcs) file
                String key_path = String.join("/", new String[]{ // build path to key (fiberset x srcs) file
                        sim_dir, "potentials", "key.dat"
                });

                // load key (fiberset x srcs) file
                File f_key = new File(key_path);
                Scanner scan_key = new Scanner(f_key);

                // save rows (number of coords) at top line... so number of lines in file is (number of coords +1)
                String products = scan_key.nextLine();
                int n_products = Integer.parseInt(products.trim());

                // pre-allocated array of doubles for products in file
                // (2 columns by default for (active_src_select,fiberset_select)
                int[][] prods = new int[n_products][2];
                int row_ind = 0;
                // assign contents of key (fiberset x srcs) file to array
                String thisLine;
                while (scan_key.hasNextLine()) { // while there are more lines to scan
                    thisLine = scan_key.nextLine();
                    String[] parts = thisLine.split("\\s+");
                    for (int i = 0; i < parts.length; i++) {
                        prods[row_ind][i] = Integer.parseInt(parts[i]);
                    }
                    row_ind++;
                }

                if (final_ve[sim_ind] == null) {
                    final_ve[sim_ind] = new double[n_products][][];
                }

                double[][][] sim_final_ve = final_ve[sim_ind];
                for (int product_ind = 0; product_ind < n_products; product_ind++) { // loop over fiberset x srcs
                    int ind_active_src_select = prods[product_ind][0];
                    int ind_fiberset_select = prods[product_ind][1];

                    Object[] src_combo_buffer = src_combo_list.getJSONArray(ind_active_src_select).toList().toArray(new Object[0]);
                    Double[] src_combo = new Double[src_combo_buffer.length];

                    for (int j = 0; j < src_combo_buffer.length; j++) {
                        if (src_combo_buffer[j].getClass() == Integer.class) {
                            src_combo[j] = ((Integer) src_combo_buffer[j]).doubleValue();
                        } else {
                            src_combo[j] = (Double) src_combo_buffer[j];
                        }
                    }

                    File f_coords = new File(String.join("/", new String[]{coord_dir, Integer.toString(ind_fiberset_select)}));
                    String[] fiber_coords_list = f_coords.list(); // create list of fiber coords (one for each fiber)
                    assert fiber_coords_list != null;
                    fiber_coords_list = Arrays.stream(fiber_coords_list).filter(s -> Pattern.matches("[0-9]+\\.dat", s)).toArray(String[]::new);

                    if (sim_final_ve[product_ind] == null) {
                        sim_final_ve[product_ind] = new double[fiber_coords_list.length][];
                    }

                    double[][] products_final_ve = sim_final_ve[product_ind];

                    for (int coords_ind = 0; coords_ind < fiber_coords_list.length; coords_ind++) { // loop over fiber coords in list of fiber coords

                        if (products_final_ve[coords_ind] == null) {
                            products_final_ve[coords_ind] = new double[bases[basis_ind][sim_ind][ind_fiberset_select][coords_ind].length];
                        }
                        double[] coords_final_ve = products_final_ve[coords_ind];

                        for (int point_ind = 0; point_ind < bases[basis_ind][sim_ind][ind_fiberset_select][coords_ind].length; point_ind++) {
                            coords_final_ve[point_ind] += bases[basis_ind][sim_ind][ind_fiberset_select][coords_ind][point_ind] * src_combo[basis_ind];
                        }
                    }
                }
            }
        }

        // WRITE FINAL VE TO FILE
        for (int sim_ind = 0; sim_ind < sims_list.length(); sim_ind++) { // loop over sims
            // get sim number for index in sims list and load sim configuration data
            int sim_num = (int) sims_list.get(sim_ind);
            String sim_dir = String.join("/", new String[]{
                    projectPath, "samples", Integer.toString(sample), "models", modelStr, "sims", Integer.toString(sim_num)
            });

            // build path to directory of fibersets
            String coord_dir = String.join("/", new String[]{
                    sim_dir, "fibersets"
            });

            // build path to directory of key (fiberset x srcs) file
            String key_path = String.join("/", new String[]{ // build path to key (fiberset x srcs) file
                    sim_dir, "potentials", "key.dat"
            });

            // load key (fiberset x srcs) file
            File f_key = new File(key_path);
            Scanner scan_key = new Scanner(f_key);

            // save rows (number of coords) at top line... so number of lines in file is (number of coords +1)
            String products = scan_key.nextLine();
            int n_products = Integer.parseInt(products.trim());

            // pre-allocated array of doubles for products in file (2 columns by default for (active_src_select,fiberset_select)
            int[][] prods = new int[n_products][2];
            int row_ind = 0;
            // assign contents of key (fiberset x srcs) file to array
            String thisLine;
            while (scan_key.hasNextLine()) { // while there are more lines to scan
                thisLine = scan_key.nextLine();
                String[] parts = thisLine.split("\\s+");
                for (int i = 0; i < parts.length; i++) {
                    prods[row_ind][i] = Integer.parseInt(parts[i]);
                }
                row_ind++;
            }

            for (int product_ind = 0; product_ind < n_products; product_ind++) { // loop over fiberset x srcs
                int ind_fiberset_select = prods[product_ind][1];
                File f_coords = new File(String.join("/", new String[]{coord_dir, Integer.toString(ind_fiberset_select)}));
                String[] fiber_coords_list = f_coords.list(); // create list of fiber coords (one for each fiber)
                assert fiber_coords_list != null;
                fiber_coords_list = Arrays.stream(fiber_coords_list).filter(s -> Pattern.matches("[0-9]+\\.dat", s)).toArray(String[]::new);

                // write Ve to file
                String ve_dir = String.join("/", new String[]{ // build path to directory of ve for each fiber coordinate
                        sim_dir, "potentials", Integer.toString(product_ind)
                });

                // if sim potentials directory does not yet exist, make it
                File vePathFile = new File(ve_dir);
                if (!vePathFile.exists()) {
                    boolean success = vePathFile.mkdirs();
                    assert success;
                }

                String src_diams_key_path = String.join("/", new String[]{coord_dir, Integer.toString(ind_fiberset_select), "diams.txt"});

                if (new File(src_diams_key_path).exists()) {
                    String dest_diams_key_path = String.join("/", new String[]{ve_dir, "diams.txt"});

                    Path src_diams_key = Paths.get(src_diams_key_path);
                    Path dest_diams_key = Paths.get(dest_diams_key_path);
                    Files.copy(src_diams_key, dest_diams_key);
                }

                for (int coords_ind = 0; coords_ind < fiber_coords_list.length; coords_ind++) { // loop over fiber coords in list of fiber coords

                    String ve_path = String.join("/", new String[]{
                            ve_dir, coords_ind + ".dat"
                    });

                    if (new File(ve_path).exists()) {
                        continue;
                    }

                    writeVe(final_ve[sim_ind][product_ind][coords_ind], ve_path);
                }
            }
        }
    }

    // https://stackoverflow.com/questions/15449711/transpose-double-matrix-with-a-java-function
    public static double[][] transposeMatrix(double [][] m){
        // pre-allocated array of doubles for transposed matrix
        double[][] temp = new double[m[0].length][m.length];

        for (int i = 0; i < m.length; i++)
            for (int j = 0; j < m[0].length; j++)
                temp[j][i] = m[i][j];
        return temp;
    }

    private static boolean writeVe(double[] ve, String ve_path) throws IOException {
        PrintWriter printWriter = new PrintWriter(ve_path);
        int len = ve.length; // number of coordinates

        // write to file: number of coordintates top line,
        // then one Ve value for each coordinate  (x,y,z) for subsequent lines
        printWriter.println(len);
        for (int i = 0; i < len; i++) {
            printWriter.println(ve[i]);
        }
        printWriter.close(); // close printWriter
        return true;
    }

    public static double[][] readCoords(String coords_path) throws FileNotFoundException {
        File f = new File(coords_path);
        Scanner scan = new Scanner(f);

        String thisLine = null;
        try {
            // save rows (number of coords) at top line... so number of lines in file is (number of coords +1)
            String rows = scan.nextLine();
            int n_rows = Integer.parseInt(rows.trim());

            // pre-allocated array of doubles for coords in file (3 columns by default for (x,y,z)
            double[][] coords = new double[n_rows][3];
            int row_ind = 0;

            // while there are more lines to scan
            while (scan.hasNextLine()) {
                thisLine = scan.nextLine();
                String[] parts = thisLine.split("\\s+");
                for(int i = 0; i < parts.length; i++) {
                    coords[row_ind][i] = Double.parseDouble(parts[i]);
                }
                row_ind++;
            }

            if (n_rows != row_ind) {
                throw new Exception("Number of coordinates (rows) in coords file " +
                        "does not match header in file: " + coords_path);
            }

            scan.close();
            return coords;
        } catch(Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    /**
     * Add all fascicles to model.
     * @return success indicator
     */
    public boolean addNerve(String sample, ModelParamGroup nerveParams, JSONObject modelData) {

        // define global nerve part names (MUST BE IDENTICAL IN Part)
        String[] fascicleTypes = new String[]{"FascicleCI", "FascicleMesh"};

        // Load configuration file
        try {
            JSONObject sampleData = JSONio.read(
                    String.join("/", new String[]{
                            this.root,
                            "samples",
                            sample,
                            "sample.json"
                    })
            );

            // Build path to fascicles
            String fasciclesPath = String.join("/", new String[]{
                    this.root,
                    "samples",
                    sample,
                    "slides",
                    "0", // these 0's are temporary (for 3d models will need to change)
                    "0",
                    "sectionwise2d",
                    "fascicles"
            });
            // Build path to nerve trace
            String nervePath = String.join("/", new String[]{
                    this.root,
                    "samples",
                    sample,
                    "slides",
                    "0", // these 0's are temporary (for 3d models will need to change)
                    "0",
                    "sectionwise2d",
                    "nerve",
                    "0"
            });

            // nerve trace filename
            HashMap<String, String[]> ndata = new HashMap<>();
            ndata.put("nerve",new String[] {"0.txt"});

            // Add epineurium
            String nerveMode = (String) sampleData.getJSONObject("modes").get("nerve");
            String reshapenerveMode = (String) sampleData.getJSONObject("modes").get("reshape_nerve");

            // backwards compatibility
            double deform_ratio = 0;
            if (sampleData.has("deform_ratio")) {
                deform_ratio = sampleData.getDouble("deform_ratio");
            } else {
                if (reshapenerveMode.equals("CIRCLE")) {
                    deform_ratio = 1;
                } else if (reshapenerveMode.equals("NONE")) {
                    deform_ratio = 0;
                }
            }

            if (nerveMode.equals("PRESENT") && !(reshapenerveMode.equals("CIRCLE") || reshapenerveMode.equals("NONE"))) {
                System.out.println("Modeling Sample with epineurium (i.e., Nerve Trace) that is not deformed toward a" +
                        "CIRCLE (or NONE) is not yet implemented");
                System.exit(0);
            }
            
            if (nerveMode.equals("PRESENT")) {
                if (deform_ratio == 1 && reshapenerveMode.equals("CIRCLE")) { //Use a circle otherwise
                    Part.createNervePartInstance("Epi_circle", 0,
                            null, this, null, sampleData, nerveParams, modelData);
                } else { //Use trace
                    Part.createNervePartInstance("Epi_trace", 0,
                            nervePath, this, ndata, sampleData, nerveParams, modelData);
                }
            }

            // Loop over all fascicle dirs
            String[] dirs = new File(fasciclesPath).list();

            JSONObject modelModes = modelData.getJSONObject("modes");  //

            if (dirs != null) {
                for (String dir: dirs) {
                    if (! dir.contains(".")) {
                        int index = Integer.parseInt(dir);
                        // Initialize data to send to Part.createPartInstance
                        HashMap<String, String[]> data = new HashMap<>();
                        // Add inners and outer files to array
                        String path = String.join("/", new String[]{fasciclesPath, dir});
                        for (String type: new String[]{"inners", "outer"}) {
                            data.put(type,
                                    new File(
                                            String.join("/", new String[]{path, type})
                                    ).list()
                            );
                        }

                        // Quick loop to make sure there are at least one of each inner and outer
                        for (String[] arr: data.values()) {
                            if (arr.length < 1) throw new IllegalStateException("There must be at least one of each inner and outer for fascicle " + index);
                        }

                        String fascicleType = null;
                        if (modelModes.has("use_ci") && !modelModes.getBoolean("use_ci")) {
                            fascicleType = fascicleTypes[1]; // "FascicleMesh"
                        } else {
                            // do "FascicleCI" if only one inner, "FascicleMesh" otherwise
                            fascicleType = data.get("inners").length == 1 ? fascicleTypes[0] : fascicleTypes[1];
                        }

                        // hand off to Part to build instance of fascicle
                        Part.createNervePartInstance(fascicleType, index, path, this,
                                data, sampleData, nerveParams, modelData);
                    }
                }
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (Exception e) {
            e.printStackTrace();
        }
        return true;
    }

    /**
     * Pre-built for-loop to iterate through all current sources in model (added in Part)
     * Can be super useful for quickly setting different currents and possibly sweeping currents
     */
    public void loopCurrents(JSONObject modelData, String projectPath, String sample, String modelStr, Boolean skipMesh, boolean pre_solve_break) throws IOException {

        long runSolStartTime = System.nanoTime();
        int index = 0;

        Set<Integer> s;
        s = this.im.currentIDs.keySet();

        for(int key_on_int = 0; key_on_int < s.size(); key_on_int++) {

            String key_on;
            String src;
            if (skipMesh){
                String key_on_int_str = Integer.toString(key_on_int + 1);
                Map key_on_obj = (Map) this.im.currentIDs.get(key_on_int_str);
                key_on = (String) key_on_obj.keySet().toArray()[0];
                src = (String) key_on_obj.get(key_on);
            } else {
                JSONObject key_on_obj = this.im.currentIDs.get(key_on_int + 1);
                key_on = (String) key_on_obj.keySet().toArray()[0];
                src = (String) key_on_obj.get(key_on);
            }

            PhysicsFeature current_on = model.physics("ec").feature(src);
            current_on.set("Qjp", 0.001); // turn on current

            String bases_directory = String.join("/", new String[]{
                    projectPath,
                    "samples",
                    sample,
                    "models",
                    modelStr,
                    "bases"
            });

            // if bases directory does not yet exist, make it
            File basesPathFile = new File(bases_directory);
            if (! basesPathFile.exists()) {
                boolean success = basesPathFile.mkdirs();
                assert success;
            }

            String mphFile = String.join("/", new String[]{
                    projectPath,
                    "samples",
                    sample,
                    "models",
                    modelStr,
                    "bases",
                    index + ".mph"
            });

            System.out.println("\tSolving electric currents for "+key_on+".");

            boolean save = true;
            if (! new File(mphFile).exists()) {
                if (!pre_solve_break) {
                    model.sol("sol1").runAll();
                    model.component("comp1").mesh("mesh1").clearMesh();
                }
                else {
                    System.out.println("\tSkipped solving for basis " +key_on+" because encountered pre_solve breakpoint. Basis MPH will be saved with no solution.");                }
            } else {
                save = false;
                System.out.println("\tSkipping solving and saving for basis " + key_on + " because found existing file: " + mphFile);
            }

            try {
                if (save) {
                    System.out.println("\tSaving MPH (mesh and solution) file to: " + mphFile);
                    model.save(mphFile);

                    File mphFileFile = new File(mphFile);

                    while(!mphFileFile.canWrite() || !mphFileFile.canRead()) {
                        System.out.println("\twaiting");
                        // wait!
                    }
                }
            } catch (IOException e) {
                e.printStackTrace();
            }

            current_on.set("Qjp", 0.000); // reset current

            index += 1;
        }

        JSONObject solution = new JSONObject();
        long estimatedRunSolTime = System.nanoTime() - runSolStartTime;
        solution.put("sol_time", estimatedRunSolTime/Math.pow(10,6)); // convert nanos to millis, this is for solving all contacts
        String version = ModelUtil.getComsolVersion(); //The getComsolVersion method returns the current COMSOL Multiphysics
        solution.put("name", version);
        modelData.put("solution", solution);
    }

    /**
     * Call only from initializer!
     * Initialize the ArrayLists in the unionContributors HashMap
     */
    public void initUnionContributors() {
        for(String unionLabel : ModelWrapper.ALL_UNIONS) {
            this.unionContributors.put(unionLabel, new ArrayList<>());
        }
    }

    /**
     * Add string id for COMSOL element to the listed unions (which have not be "created" in COMSOL yet)
     * @param contributor the string id to add (use actual id, not pseudonym)
     * @param unions which unions to add it to  (use static pseudonym constants at top of class)
     */
    public void contributeToUnions(String contributor, String[] unions) {
        for (String union: unions) {
            this.unionContributors.get(union).add(contributor);
        }
    }

    /**
     * @param union which union to from of which to get the contributors
     * @return String array of the COMSOL id's of contributors (likely ext# or csel#)
     */
    public String[] getUnionContributors(String union) {
        if (! this.unionContributors.containsKey(union)) throw new IllegalArgumentException("No such union: " + union);
        return this.unionContributors.get(union).toArray(new String[0]);
    }

    /**
     * Actually create the unions by looping through all defined ArrayLists and adding contents to a new union.
     * Will not create a union of no elements in associated ArrayList (i.e. no Peri union if only contact impedance)
     */
    public void createUnions() {
        for (String union: ModelWrapper.ALL_UNIONS) {
            String[] contributors = this.getUnionContributors(union);

            if (contributors.length > 0) {
                GeomFeature uni = model.component("comp1").geom("geom1").create(im.next("uni", union), "Union");
                uni.set("keep", true);
                uni.selection("input").set(contributors);
                uni.label(union);

                String unionCselLabel = union + "Csel";
                GeomObjectSelectionFeature csel = model.component("comp1").geom("geom1").selection().create(im.next("csel",unionCselLabel), "CumulativeSelection");
                csel.label(unionCselLabel);

                uni.set("contributeto", im.get(unionCselLabel));

            }
        }
    }

    // https://stackoverflow.com/a/29175213/11980021
    static void deleteDir(File file) {
        File[] contents = file.listFiles();
        if (contents != null) {
            for (File f : contents) {
                deleteDir(f);
            }
        }
        file.delete();
    }

    /**
     * Master procedure to run!
     * @param args
     */
    public static void main(String[] args) throws InterruptedException {
        //Load CLI args
        byte[] decodedBytes = Base64.getDecoder().decode(args[2]);
        String decodedString = new String(decodedBytes);
        JSONObject cli_args = new JSONObject(decodedString);

        // Start COMSOL Instance
        try {
            ModelUtil.connect("localhost", 2036);
        } catch(FlException e) {
            System.out.println("Could not connect to COMSOL server on port 2036, trying on port 2037...");
            try {
                ModelUtil.connect("localhost", 2037);
            } catch(FlException exc) {
                System.out.println("Could not connect to COMSOL server on port 2037, trying without specifying a port...");
                ModelUtil.connect();
            }
        }

        TimeUnit.SECONDS.sleep(5);
        ModelUtil.initStandalone(false);
        
        if (cli_args.has("comsol_progress") && cli_args.getBoolean("comsol_progress")) {
            ModelUtil.showProgress(null); // if you want to see COMSOL progress (as it makes all geometry, runs, etc.)
        }

        if (cli_args.has("comsol_progress_popup") && cli_args.getBoolean("comsol_progress_popup")) {
            ModelUtil.showProgress(true); // if you want to see COMSOL progress (as it makes all geometry, runs, etc.)
        }

        //checkout comsol license
        if (cli_args.has("wait_for_license") && !cli_args.isNull("wait_for_license")) {
            long wait_hours = cli_args.getLong("wait_for_license");
            System.out.println("Attempting to check out COMSOL license. System will wait up to " + String.valueOf(wait_hours) + " hours for an available license seat.");
            boolean lic = false;
            long start = System.currentTimeMillis();
            long stop = wait_hours * 60 * 60 * 1000 + start;
            while (System.currentTimeMillis() < stop) {
                lic = ModelUtil.checkoutLicense("COMSOL");
                if (lic == true) {
                    long now = System.currentTimeMillis();
                    double elapsed = (Long.valueOf(now).doubleValue()-Long.valueOf(start).doubleValue())/(60 * 60 * 1000);
                    System.out.printf("COMSOL license seat obtained (took %.3f hours).%n", elapsed);
                    break;
                } else {
                    TimeUnit.SECONDS.sleep(600);
                }
            }
            if (lic == false) {
                System.out.println("A COMSOL license did not become available within the specified time window. Exiting...");
                System.exit(1);
            }
        }

        // Take projectPath input to ModelWrapper and assign to string.
        String projectPath = args[0];

        // Load RUN configuration data
        String runPath = args[1]; // Take runPath input to ModelWrapper and assign to string

        JSONObject run = null;
        try {
            run = JSONio.read(runPath);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        JSONArray models_list = run.getJSONArray("models"); // get array of COMSOL models
        JSONObject break_points = new JSONObject();
        if (cli_args.has("break_point") && !cli_args.isNull("break_point")) {
            break_points.put(cli_args.getString("break_point"),true);
        }
        else if (run.has("break_points")) {
                break_points = run.getJSONObject("break_points");
        }

        Boolean endo_only_solution = false;
        if (cli_args.has("endo_only_solution") && cli_args.getBoolean("endo_only_solution")) {
            endo_only_solution = true;
        }
        else if (run.has("endo_only_solution") && run.getBoolean("endo_only_solution")) {
            endo_only_solution = true;
        }

        boolean nerve_only = false;
        boolean cuff_only = false;
        if (cli_args.has("partial_fem")  && !cli_args.isNull("partial_fem")) {
            if (cli_args.getString("partial_fem").equals("cuff_only")) {
                cuff_only = true;
            }
            else if (cli_args.getString("partial_fem").equals("nerve_only")) {
                nerve_only = true;
            }
        }
        else if (run.has("partial_fem")) {
            JSONObject partial_fem_params = run.getJSONObject("partial_fem");

            if (partial_fem_params.has("nerve_only")) {
                nerve_only = partial_fem_params.getBoolean("nerve_only");
            }

            if (partial_fem_params.has("cuff_only")) {
                cuff_only = partial_fem_params.getBoolean("cuff_only");
            }
        }


        String sample = null;
        JSONObject sampleData = null;
        String sampleFile = null;
        // Load SAMPLE configuration data
        sample = String.valueOf(Objects.requireNonNull(run).getInt("sample"));
        sampleFile = String.join("/", new String[]{"samples", sample, "sample.json"});
        try {
            sampleData = JSONio.read(projectPath + "/" + sampleFile);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        // Load mesh_dependence_model configuration data
        JSONObject meshReferenceData = null;
        try {
            meshReferenceData = JSONio.read(String.join("/", new String[]{projectPath, "config", "system", "mesh_dependent_model.json"}));
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        // variables for optimization looping
        JSONObject previousModelData = null;
        Model previousMph = null;
        IdentifierManager previousIM = null;
        HashMap<String, IdentifierManager> previousPPIMs = null;
        boolean skipMesh;

        // loop MODELS
        boolean[] models_exit_status = new boolean[models_list.length()];
        for (int model_index = 0; model_index < models_list.length(); model_index++) {

            try {
                Model model = null;
                ModelWrapper mw = null;
                skipMesh = false;

                String modelStr = String.valueOf(models_list.get(model_index));
                String bases_directory = String.join("/", new String[]{projectPath, "samples", sample, "models", modelStr, "bases"});

                System.out.println("BEGIN RUN - Model " + modelStr);

                // if bases directory does not yet exist, make it. If it exists, check that the bases are valid
                File basesPathFile = new File(bases_directory);
                boolean basesValid = true;
                if (basesPathFile.exists()) {
                    String imFile = String.join("/", new String[]{projectPath, "samples", sample, "models", modelStr, "mesh", "im.json"});
                    try {
                        JSONObject imdata = JSONio.read(imFile);
                        for (int cu=0;cu<imdata.getJSONObject("currentIDs").length();cu++) {
                            File basisFile = new File(bases_directory+"/"+String.valueOf(cu)+".mph");
                            if (!basisFile.exists()) {
                                basesValid = false;
                            }
                        }
                    }
                    catch (FileNotFoundException e) {
                        System.out.println("\tCould not validate bases because no identifier manager record exists (mesh/im.json).");
                        basesValid = false;
                    }
                } else {
                    basesValid = false;
                }

                String modelFile = null;
                if ((!basesPathFile.exists()) || (basesPathFile.list().length < 1) || (!basesValid) || nerve_only || cuff_only) {
                    // Load MODEL configuration data
                    modelFile = String.join("/", new String[]{"samples", sample, "models", modelStr, "model.json"});
                    JSONObject modelData = null;
                    try {
                        modelData = JSONio.read(projectPath + "/" + modelFile);
                    } catch (FileNotFoundException e) {
                        System.out.println("\tFailed to read MODEL config data.");
                        e.printStackTrace();
                    }

                    modelData.put("solution",JSONObject.NULL);

                    try (FileWriter file = new FileWriter("../" + modelFile)) {
                        String output = modelData.toString(2);
                        file.write(output);
                    } catch (IOException e) {
                        e.printStackTrace();
                    }

                    // if optimizing
                    boolean recycle_meshes;
                    if (run.has("recycle_meshes") && !nerve_only && !cuff_only) {
                        recycle_meshes = run.getBoolean("recycle_meshes");
                    } else {
                        recycle_meshes = false;
                    }

                    if (recycle_meshes) {
                        System.out.println("\tEntering mesh recycling logic.");
                        try {
                            ModelSearcher modelSearcher = new ModelSearcher(String.join("/", new String[]{projectPath, "samples", sample, "models"}));
                            ModelSearcher.Match meshMatch = modelSearcher.searchMeshMatch(modelData, meshReferenceData, projectPath + "/" + modelFile);

                            // if there was a mesh match
                            if (meshMatch != null) {
                                model = meshMatch.getMph();
                                mw = new ModelWrapper(model, projectPath);
                                mw.im = IdentifierManager.fromJSONObject(new JSONObject(meshMatch.getIdm().toJSONObject().toString()));
                                mw.partPrimitiveIMs = meshMatch.getPartPrimitiveIMs();

                                skipMesh = true;
                            }

                        } catch (IOException e) {
                            System.out.println("\tIssue in mesh recycling logic. Rebuilding mesh.");
                            e.printStackTrace();
                        }
                    }
                    System.out.println("\tEnd mesh recycling logic.");

                    String mediumPrimitiveString = "Medium_Primitive";
                    String instanceLabelDistalMedium = DISTAL_MEDIUM;
                    String instanceLabelProximalMedium = PROXIMAL_MEDIUM;

                    String geomFile = String.join("/", new String[]{projectPath, "samples", sample, "models", modelStr, "debug_geom.mph"});
                    String meshPath = String.join("/", new String[]{projectPath, "samples", sample, "models", modelStr, "mesh",});
                    String meshFile = String.join("/", new String[]{meshPath, "mesh.mph"});

                    // START PRE MESH
                    if (!skipMesh) {

                        System.out.println("\tRunning pre-mesh procedure.");

                        // Define model object
                        model = ModelUtil.createUnique("Model");
                        // Add component node 1
                        model.component().create("comp1", true);
                        // Add 3D geom to component node 1
                        model.component("comp1").geom().create("geom1", 3);
                        // Set default length units to micron
                        model.component("comp1").geom("geom1").lengthUnit("\u00b5m");
                        // Add materials node to component node 1
                        model.component("comp1").physics().create("ec", "ConductiveMedia", "geom1");
                        // and mesh node to component node 1
                        model.component("comp1").mesh().create("mesh1");
                        //set geometry order
                        String geometry_order;
                        try {
                            geometry_order = modelData.getJSONObject("mesh").getString("shape_order");
                        } catch (Exception e) {
                            System.out.println("\tWARNING: Invalid geometry shape order, or geometry shape order not specified. Proceeding with default order of quadratic");
                            geometry_order = "quadratic";
                        }
                        model.component("comp1").sorder(geometry_order);
                        //set solution order
                        int solution_order;
                        try {
                            solution_order = modelData.getJSONObject("solver").getInt("sorder");
                        } catch (Exception e) {
                            System.out.println("\tWARNING: Invalid solution shape order, or solution shape order not specified. Proceeding with default order of 2 (quadratic)");
                            solution_order = 2;
                        }
                        model.component("comp1").physics("ec").prop("ShapeProperty").set("order_electricpotential", solution_order);

                        // Define ModelWrapper class instance for model and projectPath
                        mw = new ModelWrapper(model, projectPath);

                        //Clear mesh stats
                        modelData.getJSONObject("mesh").put("stats", JSONObject.NULL);

                        try (FileWriter file = new FileWriter("../" + modelFile)) {
                            String output = modelData.toString(2);
                            file.write(output);
                        } catch (IOException e) {
                            e.printStackTrace();
                        }

                        // FEM MODEL GEOMETRY
                        // Set MEDIUM parameters
                        JSONObject distalMedium = modelData.getJSONObject("medium").getJSONObject("distal");
                        JSONObject proximalMedium = modelData.getJSONObject("medium").getJSONObject("proximal");
                        JSONObject meshTimes = new JSONObject();

                        String mediumParamsLabel = "Medium Parameters";
                        ModelParamGroup mediumParams = model.param().group().create(mediumParamsLabel);
                        mediumParams.label(mediumParamsLabel);

                        double proximal_length = proximalMedium.getDouble("length");
                        double proximal_radius = proximalMedium.getDouble("radius");

                        String bounds_unit = "[um]";
                        mediumParams.set("z_nerve", proximal_length + " " + bounds_unit);
                        mediumParams.set("r_proximal", proximal_radius + " " + bounds_unit);

                        if (distalMedium.getBoolean("exist")) {
                            double distal_length = distalMedium.getDouble("length");
                            double distal_radius = distalMedium.getDouble("radius");
                            double distal_x = distalMedium.getJSONObject("shift").getDouble("x");
                            double distal_y = distalMedium.getJSONObject("shift").getDouble("y");
                            double distal_z = distalMedium.getJSONObject("shift").getDouble("z");

                            mediumParams.set("z_distal", distal_length + " " + bounds_unit);
                            mediumParams.set("r_distal", distal_radius + " " + bounds_unit);
                            mediumParams.set("distal_shift_x", distal_x + " " + bounds_unit);
                            mediumParams.set("distal_shift_y", distal_y + " " + bounds_unit);
                            mediumParams.set("distal_shift_z", distal_z + " " + bounds_unit);
                        }

                        // Create PART PRIMITIVE for MEDIUM
                        String partID = mw.im.next("part", mediumPrimitiveString);
                        IdentifierManager partPrimitiveIM = null;
                        try {
                            partPrimitiveIM = Part.createEnvironmentPartPrimitive(partID, mediumPrimitiveString, mw);
                            mw.partPrimitiveIMs.put(mediumPrimitiveString, partPrimitiveIM);
                        } catch (IllegalArgumentException e) {
                            e.printStackTrace();
                        }

                        // Create PART INSTANCES for MEDIUM (Distal and Proximal)
                        if (distalMedium.getBoolean("exist")) {
                            String mediumDistal_instanceID = mw.im.next("pi", instanceLabelDistalMedium);

                            if (proximalMedium.getBoolean("distant_ground")) {
                                System.out.println("\tWARNING: you have a distal domain, as well as a proximal domain " +
                                        "that is grounded... make sure this is something you actually want to do...");
                            }

                            try {
                                Part.createEnvironmentPartInstance(mediumDistal_instanceID, instanceLabelDistalMedium, mediumPrimitiveString, mw, distalMedium);
                            } catch (IllegalArgumentException e) {
                                e.printStackTrace();
                            }
                        }

                        String mediumProximal_instanceID = mw.im.next("pi", instanceLabelProximalMedium);
                        try {
                            Part.createEnvironmentPartInstance(mediumProximal_instanceID, instanceLabelProximalMedium, mediumPrimitiveString, mw, proximalMedium);
                        } catch (IllegalArgumentException e) {
                            e.printStackTrace();
                        }

                        ModelParamGroup nerveParams = null;
                        // Set NERVE MORPHOLOGY parameters
                        JSONObject morphology = (JSONObject) sampleData.get("Morphology");
                        String morphology_unit = "um";
                        String nerveParamsLabal = "Nerve Parameters";
                        nerveParams = model.param().group().create(nerveParamsLabal);
                        nerveParams.label(nerveParamsLabal);

                        if (!cuff_only) {
                            // add NERVE (Fascicles CI/MESH and EPINEURIUM)
                            if (morphology.isNull("Nerve")) { //Monofascicle, no-epineurium case
                                nerveParams.set("a_nerve", "NaN");
                                nerveParams.set("r_nerve", modelData.getDouble("min_radius_enclosing_circle") + " [" + morphology_unit + "]");
                            } else {
                                JSONObject nerve = (JSONObject) morphology.get("Nerve");
                                nerveParams.set("a_nerve", nerve.get("area") + " [" + morphology_unit + "^2]");

                                // backwards compatibility
                                String reshapenerveMode = (String) sampleData.getJSONObject("modes").get("reshape_nerve");
                                double deform_ratio = 0;
                                if (sampleData.has("deform_ratio")) {
                                    deform_ratio = sampleData.getDouble("deform_ratio");
                                } else {
                                    if (reshapenerveMode.equals("CIRCLE")) {
                                        deform_ratio = 1;
                                    } else if (reshapenerveMode.equals("NONE")) {
                                        deform_ratio = 0;
                                    }
                                }
                                //

                                if (deform_ratio < 1) { //Use trace
                                    nerveParams.set("r_nerve", modelData.getDouble("min_radius_enclosing_circle") + " [" + morphology_unit + "]");
                                } else { //Use area of nerve
                                    nerveParams.set("r_nerve", "sqrt(a_nerve/pi)");
                                }
                            }

                            String ciCoeffsFile = String.join("/", new String[]{
                                    "config",
                                    "system",
                                    "ci_peri_thickness.json"
                            });

                            JSONObject ciCoeffsData = null;
                            try {
                                ciCoeffsData = JSONio.read(projectPath + "/" + ciCoeffsFile);
                            } catch (FileNotFoundException e) {
                                e.printStackTrace();
                            }

                            String ci_mode = sampleData.getJSONObject("modes").getString("ci_perineurium_thickness");
                            if (ci_mode.compareTo("MEASURED") != 0) {
                                JSONObject myCICoeffs = ciCoeffsData.getJSONObject("ci_perineurium_thickness_parameters").getJSONObject(ci_mode);
                                nerveParams.set("ci_a", myCICoeffs.getDouble("a") + " [micrometer/micrometer]");
                                nerveParams.set("ci_b", myCICoeffs.getDouble("b") + " [micrometer]");

                            }
                        } else {
                            nerveParams.set("a_nerve", "NaN");
                            nerveParams.set("r_nerve", modelData.getDouble("min_radius_enclosing_circle") + " [" + morphology_unit + "]");
                        }

                        if (!nerve_only) {
                            // add PART PRIMITIVES for CUFF
                            // Read cuff to build from model.json (cuff.preset) which links to JSON containing instantiations of parts
                            JSONObject cuffObject = (JSONObject) modelData.get("cuff");
                            String cuff = cuffObject.getString("preset");
                            mw.addCuffPartPrimitives(cuff);

                            // add PART INSTANCES for cuff
                            mw.addCuffPartInstances(cuff, modelData);

                            // Set CUFF POSITIONING parameters
                            String cuffConformationParamsLabel = "Cuff Conformation Parameters";
                            ModelParamGroup cuffConformationParams = model.param().group().create(cuffConformationParamsLabel);
                            cuffConformationParams.label(cuffConformationParamsLabel);

                            String cuff_shift_unit = "[micrometer]";
                            String cuff_rot_unit = "[degree]";
                            Double cuff_shift_x = modelData.getJSONObject("cuff").getJSONObject("shift").getDouble("x");
                            Double cuff_shift_y = modelData.getJSONObject("cuff").getJSONObject("shift").getDouble("y");
                            Double cuff_shift_z = modelData.getJSONObject("cuff").getJSONObject("shift").getDouble("z");
                            Double cuff_rot_pos = modelData.getJSONObject("cuff").getJSONObject("rotate").getDouble("pos_ang");
                            Double cuff_rot_add = modelData.getJSONObject("cuff").getJSONObject("rotate").getDouble("add_ang");

                            cuffConformationParams.set("cuff_shift_x", cuff_shift_x + " " + cuff_shift_unit);
                            cuffConformationParams.set("cuff_shift_y", cuff_shift_y + " " + cuff_shift_unit);
                            cuffConformationParams.set("cuff_shift_z", cuff_shift_z + " " + cuff_shift_unit);
                            cuffConformationParams.set("cuff_rot", cuff_rot_pos + cuff_rot_add + " " + cuff_rot_unit);
                        }

                        if (!cuff_only) {
                            model.nodeGroup().create(mw.im.next("grp","Contact Impedances"), "Physics", "ec");
                            model.nodeGroup(mw.im.get("Contact Impedances")).label("Contact Impedances");
                            // there are no primitives/instances for nerve parts, just build them
                            mw.addNerve(sample, nerveParams, modelData);
                        }

                        // create UNIONS
                        mw.createUnions();

                        // Saved model pre-run geometry for debugging
                        try {
                            System.out.println("\tSaving MPH (pre-geom_run) file to: " + geomFile);
                            model.save(geomFile);
                        } catch (IOException e) {
                            e.printStackTrace();
                        }

                        // break point "pre_geom_run"
                        boolean pre_geom_run;
                        if (break_points.has("pre_geom_run")) {
                            pre_geom_run = break_points.getBoolean("pre_geom_run");
                        } else {
                            pre_geom_run = false;
                        }

                        if (pre_geom_run) {
                            models_exit_status[model_index] = false;
                            System.out.println("\tpre_geom_run is the first break point encountered, moving on with next model index");
                            continue;
                        }

                        // BUILD GEOMETRY
                        System.out.println("\tBuilding the FEM geometry.");

                        try {
                            model.component("comp1").geom("geom1").run("fin");
                        } catch (Exception e) {
                            System.out.println("\tFailed to run geometry for Model Index " + modelStr + ", continuing " +
                                    "to any remaining Models");
                            e.printStackTrace();
                            continue;
                        }

                        // Saved model post-run geometry for debugging
                        try {
                            System.out.println("\tSaving MPH (post-geom_run) file to: " + geomFile);
                            model.save(geomFile);
                        } catch (IOException e) {
                            e.printStackTrace();
                        }

                        // break point "post_geom_run"
                        boolean post_geom_run;
                        if (break_points.has("post_geom_run")) {
                            post_geom_run = break_points.getBoolean("post_geom_run");
                        } else {
                            post_geom_run = false;
                        }

                        if (post_geom_run || nerve_only || cuff_only) {
                            models_exit_status[model_index] = false;
                            System.out.println("\tpost_geom_run is the first break point encountered, moving on with next model index");
                            continue;
                        }

                        // MESH
                        // ensure that the path for mesh files can be created
                        File meshPathFile = new File(meshPath);
                        if (!meshPathFile.exists()) {
                            boolean success = meshPathFile.mkdirs();
                            assert success;
                        }

                        // ditto for ppims
                        System.out.println("\tCreating PPIM dirs");
                        String ppimPath = meshPath + "/ppim";
                        File ppimPathFile = new File(ppimPath);
                        if (!ppimPathFile.exists()) {
                            boolean success = ppimPathFile.mkdirs();
                            assert success;
                        }

                        // define MESH for PROXIMAL
                        // swept: name (Sweep) and im (swe), facemethod (tri)
                        // free triangular: name (FreeTet) and im (ftet)
                        JSONObject proximalMeshParams = modelData.getJSONObject("mesh").getJSONObject("proximal");
                        String meshProximalLabel = "Mesh Proximal";
                        String meshProximalKey = proximalMeshParams.getJSONObject("type").getString("im");
                        String meshProximalName = proximalMeshParams.getJSONObject("type").getString("name");
                        MeshFeature meshProximal = model.component("comp1").mesh("mesh1").create(mw.im.next(meshProximalKey, meshProximalLabel), meshProximalName);
                        meshProximal.selection().geom("geom1", 3);
                        meshProximal.selection().named("geom1" + "_" + mediumProximal_instanceID + "_" + partPrimitiveIM.get("MEDIUM") + "_dom");

                        // if using a swept mesh, you need to define the face method
                        if (meshProximalKey.equals("swe")) {
                            String meshProximalFace = proximalMeshParams.getJSONObject("type").getString("facemethod"); // (tri)
                            meshProximal.set("facemethod", meshProximalFace);
                        }
                        meshProximal.label(meshProximalLabel);

                        String meshProximalSizeInfoLabel = "Mesh Proximal Size Info";
                        MeshFeature meshProximalSizeInfo = meshProximal.create(mw.im.next("size", meshProximalSizeInfoLabel), "Size");
                        meshProximalSizeInfo.label(meshProximalSizeInfoLabel);
                        meshProximalSizeInfo.set("custom", true);
                        meshProximalSizeInfo.set("hmaxactive", true);
                        meshProximalSizeInfo.set("hmax", proximalMeshParams.getDouble("hmax"));
                        meshProximalSizeInfo.set("hminactive", true);
                        meshProximalSizeInfo.set("hmin", proximalMeshParams.getDouble("hmin"));
                        meshProximalSizeInfo.set("hgradactive", true);
                        meshProximalSizeInfo.set("hgrad", proximalMeshParams.getDouble("hgrad"));
                        meshProximalSizeInfo.set("hcurveactive", true);
                        meshProximalSizeInfo.set("hcurve", proximalMeshParams.getDouble("hcurve"));
                        meshProximalSizeInfo.set("hnarrowactive", true);
                        meshProximalSizeInfo.set("hnarrow", proximalMeshParams.getDouble("hnarrow"));

                        // Saved model pre-mesh for debugging
                        try {
                            System.out.println("\tSaving MPH (pre-proximal mesh) file to: " + meshFile);
                            model.save(meshFile);
                        } catch (IOException e) {
                            System.out.println("\tFailed to save geometry for Model Index " + modelStr + ", continuing " +
                                    "to any remaining Models");
                            e.printStackTrace();
                            continue;
                        }

                        // break point "pre_mesh_proximal"
                        boolean pre_mesh_proximal;
                        if (break_points.has("pre_mesh_proximal")) {
                            pre_mesh_proximal = break_points.getBoolean("pre_mesh_proximal");
                        } else {
                            pre_mesh_proximal = false;
                        }

                        if (pre_mesh_proximal) {
                            models_exit_status[model_index] = false;
                            System.out.println("\tpre_mesh_proximal is the first break point encountered, moving on with next model index");
                            continue;
                        }

                        System.out.println("\tMeshing proximal parts... will take a while");

                        long proximalMeshStartTime = System.nanoTime();
                         try {
                             model.component("comp1").mesh("mesh1").run(mw.im.get(meshProximalLabel));
                         } catch (Exception e) {
                             System.out.println("\tFailed to mesh proximal geometry for Model Index " + modelStr +
                                     ", continuing to any remaining Models");
                             e.printStackTrace();
                             continue;
                         }

                        long estimatedProximalMeshTime = System.nanoTime() - proximalMeshStartTime;
                        meshTimes.put("proximal", estimatedProximalMeshTime / Math.pow(10, 6)); // convert nanos to millis

                        // put nerve to mesh, rest to mesh, mesh to modelData
                        JSONObject mesh = modelData.getJSONObject("mesh");

                        TimeUnit.SECONDS.sleep(1);

                        // Saved model pre-mesh for debugging
                        model.save(meshFile);
                        TimeUnit.SECONDS.sleep(5);

                        // break point "post_mesh_proximal"
                        boolean post_mesh_proximal;
                        if (break_points.has("post_mesh_proximal")) {
                            post_mesh_proximal = break_points.getBoolean("post_mesh_proximal");
                        } else {
                            post_mesh_proximal = false;
                        }

                        if (post_mesh_proximal) {
                            models_exit_status[model_index] = false;
                            System.out.println("\tpost_mesh_proximal is the first break point encountered, moving on with next model index");
                            continue;
                        }

                        // define MESH for DISTAL
                        // swept: name (Sweep) and im (swe), facemethod (tri)
                        // free triangular: name (FreeTet) and im (ftet)
                        if (distalMedium.getBoolean("exist")) {
                            String meshDistalLabel = "Mesh Distal";
                            JSONObject distalMeshParams = modelData.getJSONObject("mesh").getJSONObject("distal");
                            String meshDistalKey = distalMeshParams.getJSONObject("type").getString("im");
                            String meshDistalName = distalMeshParams.getJSONObject("type").getString("name");
                            MeshFeature meshDistal = model.component("comp1").mesh("mesh1").create(mw.im.next(meshDistalKey, meshDistalLabel), meshDistalName);
                            meshDistal.selection().geom("geom1", 3);
                            meshDistal.selection().remaining();
                            meshDistal.label(meshDistalLabel);

                            String meshDistalSizeInfoLabel = "Mesh Distal Size Info";
                            MeshFeature meshDistalSizeInfo = meshDistal.create(mw.im.next("size", meshDistalSizeInfoLabel), "Size");
                            meshDistalSizeInfo.label(meshDistalSizeInfoLabel);

                            meshDistalSizeInfo.set("custom", true);
                            meshDistalSizeInfo.set("hmaxactive", true);
                            meshDistalSizeInfo.set("hmax", distalMeshParams.getDouble("hmax"));
                            meshDistalSizeInfo.set("hminactive", true);
                            meshDistalSizeInfo.set("hmin", distalMeshParams.getDouble("hmin"));
                            meshDistalSizeInfo.set("hgradactive", true);
                            meshDistalSizeInfo.set("hgrad", distalMeshParams.getDouble("hgrad"));
                            meshDistalSizeInfo.set("hcurveactive", true);
                            meshDistalSizeInfo.set("hcurve", distalMeshParams.getDouble("hcurve"));
                            meshDistalSizeInfo.set("hnarrowactive", true);
                            meshDistalSizeInfo.set("hnarrow", distalMeshParams.getDouble("hnarrow"));

                            // Saved model pre-mesh for debugging
                            try {
                                System.out.println("\tSaving MPH (pre-distal mesh) file to: " + meshFile);
                                model.save(meshFile);
                            } catch (IOException e) {
                                System.out.println("\tFailed to save geometry for Model Index " + modelStr + ", continuing " +
                                        "to any remaining Models");
                                e.printStackTrace();
                                continue;
                            }

                            // break point "pre_mesh_distal"
                            boolean pre_mesh_distal;
                            if (break_points.has("pre_mesh_distal")) {
                                pre_mesh_distal = break_points.getBoolean("pre_mesh_distal");
                            } else {
                                pre_mesh_distal = false;
                            }

                            if (pre_mesh_distal) {
                                models_exit_status[model_index] = false;
                                System.out.println("\tpre_mesh_distal is the first break point encountered, moving on with next model index");
                                continue;
                            }

                            System.out.println("\tMeshing the distal parts... will take a while");
                            long distalMeshStartTime = System.nanoTime();
                            try {
                                model.component("comp1").mesh("mesh1").run(mw.im.get(meshDistalLabel));
                            } catch (Exception e) {
                                System.out.println("\tFailed to mesh distal geometry for Model Index " + modelStr +
                                        ", continuing to any remaining Models");
                                e.printStackTrace();
                                continue;
                            }
                            long estimatedRestMeshTime = System.nanoTime() - distalMeshStartTime;
                            meshTimes.put("distal", estimatedRestMeshTime / Math.pow(10, 6)); // convert nanos to millis

                            // Saved model post-mesh distal for debugging
                            try {
                                System.out.println("\tSaving MPH (post-distal mesh) file to: " + meshFile);
                                model.save(meshFile);
                            } catch (IOException e) {
                                System.out.println("\tFailed to save geometry for Model Index " + modelStr + ", continuing " +
                                        "to any remaining Models");
                                e.printStackTrace();
                                continue;
                            }

                            // break point "post_mesh_distal"
                            boolean post_mesh_distal;
                            if (break_points.has("post_mesh_distal")) {
                                post_mesh_distal = break_points.getBoolean("post_mesh_distal");
                            } else {
                                post_mesh_distal = false;
                            }

                            if (post_mesh_distal) {
                                models_exit_status[model_index] = false;
                                System.out.println("\tpost_mesh_distal is the first break point encountered, moving on with next model index");
                                continue;
                            }
                        }

                        System.out.println("\tSaving mesh statistics.");

                        // MESH STATISTICS
                        String quality_measure;
                        if (modelData.getJSONObject("mesh").has("quality_measure")) {
                            quality_measure = modelData.getJSONObject("mesh").getString("quality_measure");
                        }
                        else {
                            quality_measure = "vollength";
                            System.out.println("\tNo quality measure for mesh, using default (vollength)");
                        }

                        model.component("comp1").mesh("mesh1").stat().setQualityMeasure(quality_measure);
                            // could use: skewness, maxangle, volcircum, vollength, condition, growth...

                        Integer number_elements = model.component("comp1").mesh("mesh1").getNumElem("all");
                        Double min_quality = model.component("comp1").mesh("mesh1").getMinQuality("all");
                        Double mean_quality = model.component("comp1").mesh("mesh1").getMeanQuality("all");
                        Double min_volume = model.component("comp1").mesh("mesh1").getMinVolume("all");
                        Double volume = model.component("comp1").mesh("mesh1").getVolume("all");

                        JSONObject meshStats = new JSONObject();
                        meshStats.put("mesh_times",meshTimes);
                        meshStats.put("number_elements", number_elements);
                        meshStats.put("min_quality", min_quality);
                        meshStats.put("mean_quality", mean_quality);
                        meshStats.put("mean_quality", mean_quality);
                        meshStats.put("min_volume", min_volume);
                        meshStats.put("volume", volume);
                        meshStats.put("quality_measure_used", quality_measure);
                        meshStats.put("name", ModelUtil.getComsolVersion());
                        mesh.put("stats", meshStats);
                        modelData.put("mesh", mesh);

                        try (FileWriter file = new FileWriter("../" + modelFile)) {
                            String output = modelData.toString(2);
                            file.write(output);
                        } catch (IOException e) {
                            e.printStackTrace();
                        }

                        System.out.println("\tDONE MESHING");

                        try {
                            // save mesh.mph
                            System.out.println("\tSaving MPH (post-mesh) file to: " + meshFile);
                            model.save(meshFile);
                        } catch (IOException e) {
                            System.out.println("\tFailed to save mesh.mph file for Model Index " + modelStr +
                                    ", continuing to any remaining Models");
                            e.printStackTrace();
                        }

                        String imFile = String.join("/", new String[]{projectPath, "samples", sample, "models", modelStr, "mesh", "im.json"});

                        // save IM !!!!
                        JSONio.write(imFile, mw.im.toJSONObject()); // write to file

                        // save ppIMs !!!!
                        for (String name : mw.partPrimitiveIMs.keySet()) {
                            JSONio.write(ppimPath + "/" + name + ".json", mw.partPrimitiveIMs.get(name).toJSONObject());
                        }

                        boolean keep_debug_geom;
                        if (run.has("keep") && run.getJSONObject("keep").has("debug_geom")) {
                            keep_debug_geom = run.getJSONObject("keep").getBoolean("debug_geom");
                        } else {
                            keep_debug_geom = true;
                        }

                        if (!keep_debug_geom) {
                            File debug_geom_file = new File(geomFile);
                            debug_geom_file.delete();
                            System.out.println("\tSuccessfully saved mesh.mph and ppim's, therefore deleted debug_geom.mph file.");
                        }

                    }

                    //////////////// START POST MESH
                    // IMPORTANT THAT MODEL IS NOT NULL HERE!!
                    assert model != null;
                    assert sampleData != null;

                    // add MATERIAL DEFINITIONS
                    String materialParamsLabel = "Material Parameters";
                    ModelParamGroup materialParams = model.param().group().create(materialParamsLabel);
                    materialParams.label(materialParamsLabel);

                    String nerveMode = (String) sampleData.getJSONObject("modes").get("nerve");
                    ArrayList<String> bio_materials = new ArrayList<>(Arrays.asList("medium", "perineurium", "endoneurium"));
                    if (nerveMode.equals("PRESENT")) {
                        bio_materials.add("epineurium");
                    }
                    mw.addMaterialDefinitions(bio_materials, modelData, materialParams);

                    JSONObject cuffObject = (JSONObject) modelData.get("cuff");
                    String cuff = cuffObject.getString("preset");

                    JSONObject cuffData = JSONio.read(String.join("/",
                            new String[]{mw.root, "config", "system", "cuffs", cuff}));

                    ArrayList<String> cuff_materials = new ArrayList<>();
                    // loop through all part instances
                    for (Object item : (JSONArray) cuffData.get("instances")) {
                        JSONObject itemObject = (JSONObject) item;
                        for (Object function : itemObject.getJSONArray("materials")) {
                            JSONObject functionObject = (JSONObject) function;
                            cuff_materials.add(functionObject.getString("info"));
                        }
                    }
                    mw.addMaterialDefinitions(cuff_materials, modelData, materialParams);

                    // Add material assignments (links)
                    // DOMAIN
                    JSONObject distalMedium = modelData.getJSONObject("medium").getJSONObject("distal");
                    String mediumMaterial = mw.im.get("medium");
                    IdentifierManager myIM = mw.getPartPrimitiveIM(mediumPrimitiveString);
                    if (myIM == null)
                        throw new IllegalArgumentException("IdentifierManager not created for name: " + mediumPrimitiveString);
                    String[] myLabels = myIM.labels; // may be null, but that is ok if not used
                    String selection = myLabels[0];

                    if (distalMedium.getBoolean("exist")) {

                        String linkLabel = String.join("/", new String[]{instanceLabelDistalMedium, selection, "medium"});
                        Material mat = model.component("comp1").material().create(mw.im.next("matlnk", linkLabel), "Link");
                        mat.label(linkLabel);
                        mat.set("link", mediumMaterial);
                        mat.selection().named("geom1_" + mw.im.get(instanceLabelDistalMedium) + "_" + myIM.get(selection) + "_dom");
                    } else {

                        String linkLabel = String.join("/", new String[]{instanceLabelProximalMedium, selection, "medium"});
                        Material mat = model.component("comp1").material().create(mw.im.next("matlnk", linkLabel), "Link");
                        mat.label(linkLabel);
                        mat.set("link", mediumMaterial);
                        mat.selection().named("geom1_" + mw.im.get(instanceLabelProximalMedium) + "_" + myIM.get(selection) + "_dom");
                    }

                    // CUFF
                    mw.addCuffPartMaterialAssignments(cuffData);

                    // NERVE
                    // Add epineurium only if NerveMode == PRESENT
                    if (nerveMode.equals("PRESENT")) {
                        String epineuriumMatLinkLabel = "epineurium material";
                        PropFeature epineuriumMatLink = model.component("comp1").material().create(mw.im.next("matlnk", epineuriumMatLinkLabel), "Link");
                        epineuriumMatLink.selection().named("geom1" + "_" + mw.im.get("EPINEURIUM") + "_dom");
                        epineuriumMatLink.label(epineuriumMatLinkLabel);
                        epineuriumMatLink.set("link", mw.im.get("epineurium"));
                    }

                    // Add perineurium material only if there are any fascicles being meshed
                    if (mw.im.get("periUnionCsel") != null) {
                        String perineuriumMatLinkLabel = "perineurium material";
                        PropFeature perineuriumMatLink = model.component("comp1").material().create(mw.im.next("matlnk", perineuriumMatLinkLabel), "Link");
                        perineuriumMatLink.selection().named("geom1" + "_" + mw.im.get("periUnionCsel") + "_dom");
                        perineuriumMatLink.label(perineuriumMatLinkLabel);
                        perineuriumMatLink.set("link", mw.im.get("perineurium"));
                    }

                    // Will always need to add endoneurium material
                    String fascicleMatLinkLabel = "endoneurium material";
                    PropFeature fascicleMatLink = model.component("comp1").material().create(mw.im.next("matlnk", fascicleMatLinkLabel), "Link");
                    fascicleMatLink.selection().named("geom1" + "_" + mw.im.get("endoUnionCsel") + "_dom");
                    fascicleMatLink.label(fascicleMatLinkLabel);
                    fascicleMatLink.set("link", mw.im.get("endoneurium"));

                    // break point "post_mesh_distal"
                    boolean post_material_assign;
                    if (break_points.has("post_material_assign")) {
                        post_material_assign = break_points.getBoolean("post_material_assign");
                    } else {
                        post_material_assign = false;
                    }

                    if (post_material_assign) {
                        models_exit_status[model_index] = false;
                        System.out.println("\tpost_material_assign is the first break point encountered, moving on with next model index");
                        continue;
                    }

                    // Solve
                    model.study().create("std1");
                    model.study("std1").setGenConv(true);
                    model.study("std1").create("stat", "Stationary");
                    model.study("std1").feature("stat").activate("ec", true);

                    model.sol().create("sol1");
                    model.sol("sol1").study("std1");

                    model.study("std1").feature("stat").set("notlistsolnum", 1);
                    model.study("std1").feature("stat").set("notsolnum", "1");
                    model.study("std1").feature("stat").set("listsolnum", 1);
                    model.study("std1").feature("stat").set("solnum", "1");
                    if (endo_only_solution) {
                        model.study("std1").feature("stat").set("usestoresel", "selection");
                        model.study("std1").feature("stat").set("storesel", new String[]{"geom1_" + mw.im.get("endoUnionCsel") + "_dom"});
                    }

                    model.sol("sol1").create("st1", "StudyStep");
                    model.sol("sol1").feature("st1").set("study", "std1");
                    model.sol("sol1").feature("st1").set("studystep", "stat");
                    model.sol("sol1").create("v1", "Variables");
                    model.sol("sol1").feature("v1").set("control", "stat");

                    model.sol("sol1").create("s1", "Stationary");
                    model.sol("sol1").feature("s1").create("fc1", "FullyCoupled");
                    model.sol("sol1").feature("s1").create("i1", "Iterative");
                    model.sol("sol1").feature("s1").feature("i1").set("linsolver", "cg");
                    model.sol("sol1").feature("s1").feature("i1").create("mg1", "Multigrid");
                    model.sol("sol1").feature("s1").feature("i1").feature("mg1").set("prefun", "amg");
                    model.sol("sol1").feature("s1").feature("fc1").set("linsolver", "i1");
                    model.sol("sol1").feature("s1").feature().remove("fcDef");
                    model.sol("sol1").attach("std1");

                    // break point "post_mesh_distal"
                    boolean pre_loop_currents;
                    if (break_points.has("pre_loop_currents")) {
                        pre_loop_currents = break_points.getBoolean("pre_loop_currents");
                    } else {
                        pre_loop_currents = false;
                    }

                    if (pre_loop_currents) {
                        models_exit_status[model_index] = false;
                        System.out.println("\tpre_loop_currents is the first break point encountered, moving on with next model index");
                        continue;
                    }

                    // break point "post_mesh_distal"
                    boolean pre_solve_break;
                    if (break_points.has("pre_solve")) {
                        pre_solve_break = break_points.getBoolean("pre_solve");
                    } else {
                        pre_solve_break = false;
                    }

                    mw.loopCurrents(modelData, projectPath, sample, modelStr, skipMesh, pre_solve_break);

                    if (pre_solve_break) {
                        models_exit_status[model_index] = false;
                        System.out.println("\tpre_solve is the first break point encountered, moving on with next model index\n");
                        continue;
                    }

                    ModelUtil.remove(model.tag());

                    boolean keep_mesh;
                    if (run.has("keep") && run.getJSONObject("keep").has("mesh")) {
                        keep_mesh = run.getJSONObject("keep").getBoolean("mesh");
                    } else {
                        keep_mesh = true;
                    }

                    if (!keep_mesh) {
                        File mesh_path = new File(meshPath);
                        deleteDir(mesh_path);
                        System.out.println("\tSuccessfully solved for /bases, therefore deleted /mesh directory.");
                    }

                    try (FileWriter file = new FileWriter("../" + modelFile)) {
                        String output = modelData.toString(2);
                        file.write(output);
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }

                // If no Sim configs, SKIP
                JSONArray sims_list = run.getJSONArray("sims");
                if (sims_list.length() >= 1) {
                    try {
                        extractAllPotentials(projectPath, runPath, modelStr);
                    } catch (Exception e) {
                        System.out.println("\tFailed to extract potentials for Model Index " + modelStr +
                                ", continuing to any remaining Models");
                        e.printStackTrace();
                        continue;
                    }
                }

                String model_path = String.join("/", new String[]{ // build path to sim config file
                        projectPath, "samples", sample, "models", modelStr
                });

                // construct bases path (to MPH)
                String basesPath = String.join("/", new String[]{
                        model_path, "bases"
                });

                boolean keep_bases;
                if (run.has("keep") && run.getJSONObject("keep").has("bases")) {
                    keep_bases = run.getJSONObject("keep").getBoolean("bases");
                } else {
                    keep_bases = true;
                }

                if (!keep_bases) {
                    File bases_path = new File(basesPath);
                    deleteDir(bases_path);
                    System.out.println("\tSuccessfully extracted potentials, therefore deleted /bases directory.");
                }

                models_exit_status[model_index] = true;
            } catch (Exception e) {
                models_exit_status[model_index] = false;
                System.out.println("\tFailed to mesh/solve/extract potentials for model " + models_list.get(model_index));
                e.printStackTrace();
            }
        }
        // keep track of successful and failed model indices, continue in Python for successes only

        ModelUtil.disconnect();
        System.out.println("Disconnected from COMSOL Server");
        run.put("models_exit_status", models_exit_status);

        try (FileWriter file = new FileWriter(runPath)) {
            String output = run.toString(2);
            file.write(output);

        } catch (IOException e) {
            e.printStackTrace();
        }

        System.exit(0);
    }
}
