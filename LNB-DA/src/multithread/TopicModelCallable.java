package multithread;

import java.util.concurrent.Callable;

import topicmodel.ModelPrinter;
import topicmodel.TopicModel;
import topicmodel.TopicModelParameters;
import nlp.Corpus;

public class TopicModelCallable implements Callable<TopicModel> {
	private TopicModelParameters param = null;
	private Corpus corpus = null;

	public TopicModelCallable(Corpus corpus2, TopicModelParameters param2) {
		corpus = corpus2;
		param = param2;
	}

	@Override
	/**
	 * Run the topic model in a domain and print it into the disk.
	 */
	public TopicModel call() throws Exception {
		System.out.println("\"" + param.domain + "\" <" + param.modelName
				+ "> Starts...");

		TopicModel model = TopicModel.selectModel(corpus, param);
		model.run();

		ModelPrinter modelPrinter = new ModelPrinter(model);
		modelPrinter.printModel(param.outputModelDirectory);

		System.out.println("\"" + param.domain + "\" <" + param.modelName
				+ "> Ends!");

		return model;
	}
}
