using System;
using System.Collections.Generic;
using System.Threading.Tasks;
namespace OxyBrain
{
	public class Neuron
	{
		private static uint count;

		private uint id {
			get;
		}

		private List<int> inputsId = new List<int> ();

		public Neuron(){
			id = count;
			count++;
		}

		public double output {
			get;
			private set;
		} = 0.0;

		public double delta {
			get;
			private set;
		} = 0.00;

		public List<double> importances {
			private set;
			get;
		} = new List<double>();
		private List<double> inputs = new List<double>();

		private double fi()
		{
			return Math.Pow (Math.E, -this.output) / Math.Pow (Math.Pow (Math.E, this.output) + 1, 2);
		}

		public double Delta(double step, double factOutput, double importance, double input)
		{
			this.delta = step * (factOutput - this.output) * this.fi () * input;
			return delta;
		}

		private double Backpropagation(double step, List<double> nextLayerDelta, List<double> nextLayerImportances, double input)
		{
			double sum = 0.00;
			try
			{
				for(int i = 0; i < nextLayerDelta.Count - 1; i++)
					{
						Console.WriteLine("Next Layer Delta: " + nextLayerDelta [i] + " Next Layer Importances: "
						+ nextLayerImportances [i]);
						nextLayerDelta [i] *= nextLayerImportances [i];	
						sum += nextLayerDelta [i];
						Console.WriteLine("SUM: " + sum);
					}
			}
			catch (System.Exception)
			{
				Console.WriteLine("Delta: " + nextLayerDelta.Count + " Importances: " + nextLayerImportances.Count);
				throw;
			}
			
			Console.WriteLine("OUT: " + step * sum * this.fi () * input);
			return step * sum * this.fi () * input;
		}

		public double activeFunction(double sum)
		{
			return 1 / (1 + Math.Pow (Math.E, -sum));
		}

		public double Active(List<double> data, Func<List<double>, List<double>, double> average)
		{
			inputs = data;
			if(importances.Count != data.Count)
			{
				var rnd = new Random();
				for (int i = 0; i < data.Count; i++) 
				{
					importances.Add (rnd.NextDouble ());
				}
			}

			double sum = average (data, importances);
			output = activeFunction (sum);
			return output;
		}

		public void	learn(double step, double o )
		{
			for (int i = 0; i < importances.Count; i++) 
			{
				importances [i] += Delta (step, o, importances [i], inputs [i]);
			}
		}

		public void learn(double step, List<double> nextLayearDelta, List<double> nextLayearImportances)
		{
			Parallel.For (0, importances.Count,
			 new ParallelOptions (){ MaxDegreeOfParallelism = Environment.ProcessorCount }, 
			 i => importances [i] += Backpropagation (step, nextLayearDelta, nextLayearImportances, inputs [i]));
		}

	}
}

