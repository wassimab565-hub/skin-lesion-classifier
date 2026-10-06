import { HttpClient } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable } from 'rxjs';
import { Prediction } from '../models/prediction.model';

const API_URL = 'http://127.0.0.1:8000';

@Injectable({ providedIn: 'root' })
export class PredictionService {
  private http = inject(HttpClient);

  predict(file: File): Observable<Prediction> {
    const form = new FormData();
    form.append('file', file);
    return this.http.post<Prediction>(`${API_URL}/predict`, form);
  }
}