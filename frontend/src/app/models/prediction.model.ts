export interface ClassProbability {
  label: string;
  name: string;
  probability: number;
}

export interface Prediction {
  predicted_class: string;
  predicted_name: string;
  confidence: number;
  probabilities: ClassProbability[];
  gradcam_image: string;
  model_name: string;
  input_size: string;
  inference_time_ms: number;
}