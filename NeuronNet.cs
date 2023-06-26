using System;
using System.Threading.Tasks;
using OxyBrain;
using System.Collections.Generic;
using System.Runtime.Serialization.Formatters.Binary;

namespace OxyBrain
{
	public class NeuronNet
	{
		private List<List<Neuron>> net = new List<List<Neuron>>();

		public NeuronNet (int  inputCount, List<int> layerInfo, int outputCount)
		{
			List<Neuron> data = new List<Neuron>();
			for(int i = 0; i < inputCount; i++ )
			{
				data.Add (new Neuron ());
			}

			net.Add (data);
			data = new List<Neuron> ();

			foreach (var item in layerInfo) 
			{
				for (int i = 0; i < item; i++) 
				{
					data.Add (new Neuron ());
				}
				net.Add (data);
				data = new List<Neuron>();
			}

			for(int i = 0; i < outputCount; i++ )
			{
				data.Add (new Neuron ());
			}

			net.Add (data);
			data = new List<Neuron>();
		}

		public List<double> getLayerDeltas(List<Neuron> neurons)
		{
			List<double> data = new List<double>();
			foreach (var item in neurons) 
			{
				data.Add (item.delta);
			}
			return data;
		}

		public List<double> getLayerNextImportances(List<Neuron> neurons)
		{
			List<double> data  = new List<double>();
			for (int i = 1; i < neurons.Count; i++)
			{
				var nImportances = neurons[i].importances;
				foreach(var item in nImportances){
					data.Add(item);
				}
	
			}
				
			return data;
		}
			
		
		public void learn(List <double> output)
		{
			List <Neuron> endlayer = net [net.Count - 1];
			
			Parallel.For (0, endlayer.Count,
			 new ParallelOptions (){ MaxDegreeOfParallelism = Environment.ProcessorCount }, 
			 i => endlayer[i].learn (1, output [i]));

			for (int x = 1; x < net.Count; x++) 
			{
				var layer = net [net.Count - x];

				 for (int j = 0; j < layer.Count-1; j++)
				 {

					 layer[j].learn(1,getLayerDeltas(layer), getLayerNextImportances(layer));
				 }
			}
		}

		public double WeightedAverage(List<double> data , List<double> importances)
		{
			double avanrage = 0;
			for (int i = 0; i < data.Count; i++) 
			{
				avanrage += data [i] * importances [i];
			}
			avanrage /= data.Count;
			return avanrage;
		}

		public List<double> run(List <double> input)
		{
			List <double> outputs = new List<double>();
			foreach(var Layer in net)
			{
				outputs = new List<double>();

				Parallel.ForEach (Layer, new ParallelOptions(){MaxDegreeOfParallelism = Environment.ProcessorCount}, neuron => 
				{
						neuron.Active(input, WeightedAverage);
				});

				//foreach(var neuron in Layer)
				//	neuron.Active(input, WeightedAverage);
				
				foreach (var neuron in Layer) 
				{
					outputs.Add (neuron.output);
				}
				input = outputs;
			}
			return outputs;
		}
	}
}

