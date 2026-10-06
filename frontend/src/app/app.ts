import { Component, inject, signal } from '@angular/core';
import { Prediction } from './models/prediction.model';
import { PredictionService } from './services/prediction.service';

@Component({
  selector: 'app-root',
  templateUrl: './app.html',
  styleUrl: './app.css',
})
export class App {
  private service = inject(PredictionService);

  file = signal<File | null>(null);
  previewUrl = signal<string | null>(null);
  result = signal<Prediction | null>(null);
  analyzedAt = signal('');
  loading = signal(false);
  error = signal<string | null>(null);
  dragging = signal(false);

  private readonly maxSize = 10 * 1024 * 1024;
  private readonly allowed = ['image/jpeg', 'image/png'];

  onFileInput(event: Event) {
    const input = event.target as HTMLInputElement;
    if (input.files?.length) this.setFile(input.files[0]);
    input.value = '';
  }

  onDrop(event: DragEvent) {
    event.preventDefault();
    this.dragging.set(false);
    const f = event.dataTransfer?.files?.[0];
    if (f) this.setFile(f);
  }

  onDragOver(event: DragEvent) {
    event.preventDefault();
    this.dragging.set(true);
  }

  onDragLeave() {
    this.dragging.set(false);
  }

  private setFile(f: File) {
    if (!this.allowed.includes(f.type)) {
      this.error.set('Unsupported format (JPG or PNG only).');
      return;
    }
    if (f.size > this.maxSize) {
      this.error.set('Image too large (10 MB max).');
      return;
    }
    this.reset();
    this.file.set(f);
    this.previewUrl.set(URL.createObjectURL(f));
  }

  reset() {
    const url = this.previewUrl();
    if (url) URL.revokeObjectURL(url);
    this.file.set(null);
    this.previewUrl.set(null);
    this.result.set(null);
    this.error.set(null);
  }

  analyze() {
    const f = this.file();
    if (!f) return;
    this.loading.set(true);
    this.error.set(null);
    this.result.set(null);

    this.service.predict(f).subscribe({
      next: (r) => {
        this.result.set(r);
        this.analyzedAt.set(new Date().toLocaleString('en-GB'));
        this.loading.set(false);
      },
      error: () => {
        this.error.set('Unable to reach the API. Make sure the backend is running.');
        this.loading.set(false);
      },
    });
  }

  percent(p: number): string {
    return (p * 100).toFixed(1) + '%';
  }

  seconds(ms: number): string {
    return (ms / 1000).toFixed(2) + ' s';
  }
}