/**
* @author JunSong E-mail:songjun54cm@gmail.com
* @version Create Time: Apr 13, 2018 3:07:34 PM
*/
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.TensorFlow;
import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import java.util.ArrayList;
import java.util.List;
import java.io.File;  
import java.io.InputStreamReader;  
import java.io.BufferedReader;  
import java.io.BufferedWriter;  
import java.io.FileInputStream;  
import java.io.FileWriter;
import java.io.IOException;

public class MnistClassifier {

	/**
	 * @param args
	 * @throws IOException 
	 */
	public static void main(String[] args) throws IOException {
		// TODO Auto-generated method stub
		String model_dir = "E:\\python_workspace\\test_projects\\tensorflow_example\\output\\cpu\\saved_model\\epoch_100";
		System.out.printf("load model from %s%n", model_dir);
		// load model Bundle
		SavedModelBundle bundle = SavedModelBundle.load(model_dir, "train");
		// create session from Bundle
		Session sess = bundle.session();
		// create input Tensor
		Tensor eval_features = get_eval_features();
		Tensor eval_labels = get_eval_labels();
		// run model and get result.
		Tensor  acc = null;
		acc = sess.runner()
				.feed("input_label", eval_labels)
				.feed("input_feature", eval_features)
				.fetch("pred_accuracy")
				.run()
				.get(0);
//		System.out.println(acc.floatValue());
		System.out.printf("prediction accuracy: %f%n", acc.floatValue());
		eval_features.close();
		eval_labels.close();
		acc.close();
	}
	
	private static Tensor get_eval_features() throws IOException {
		String eval_fea_file_path = "E:\\python_workspace\\test_projects\\tensorflow_example\\data\\eval_data.txt";
		File f = new File(eval_fea_file_path);
		InputStreamReader reader = new InputStreamReader(new FileInputStream(f));
		BufferedReader br = new BufferedReader(reader);
		ArrayList<ArrayList<Float>> fea_data = new ArrayList<ArrayList<Float>>();
		String line = "";
		line = br.readLine();
		while(line != null) {
			String[] vals = line.split(" ");
			ArrayList<Float> fea_vals = new ArrayList<Float>();
			for(String val : vals) {
				fea_vals.add(Float.parseFloat(val));
			}
			fea_data.add(fea_vals);
			line = br.readLine();
		}
		int sample_num = fea_data.size();
		
		FloatBuffer buf = FloatBuffer.allocate(sample_num*28*28);
		for(ArrayList<Float> fea : fea_data) {
			for(Float v : fea) {
				buf.put(v);
			}
		}
		buf.flip();
		long[] shape = new long[] {sample_num, 28, 28, 1};
		Tensor t = Tensor.create(shape, buf);
		return t;
	}
	
	private static Tensor get_eval_labels() throws IOException {
		String label_file_path = "E:\\python_workspace\\test_projects\\tensorflow_example\\data\\eval_labels.txt";
		File f = new File(label_file_path);
		InputStreamReader reader = new InputStreamReader(new FileInputStream(f));
		BufferedReader br = new BufferedReader(reader);
		ArrayList<Long> label_data = new ArrayList<Long>();
		String line = "";
		line = br.readLine();
		while(line != null) {
			label_data.add(Long.parseLong(line));
			line = br.readLine();
		}
		int sample_num = label_data.size();
		LongBuffer buf = LongBuffer.allocate(sample_num);
		for(Long v : label_data) {
			buf.put(v);
		}
		buf.flip();
		long[] shape = new long[] {sample_num};
		Tensor t = Tensor.create(shape, buf);
		return t;
	}
}
