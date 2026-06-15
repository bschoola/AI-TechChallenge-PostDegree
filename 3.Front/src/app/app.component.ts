import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import {
  PredictionService,
  PredictionInput,
  PredictionResponse,
} from './services/prediction.service';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css',
})
export class AppComponent {
  form: PredictionInput = {
    area_pior: null!,
    textura_pior: null!,
    pontos_concavos_pior: null!,
    concavidade_pior: null!,
  };

  result: PredictionResponse | null = null;
  loading = false;
  error: string | null = null;
  dataAtual = new Date();

  constructor(private predictionService: PredictionService) {}

  submit(): void {
    this.loading = true;
    this.error = null;
    this.result = null;
    this.dataAtual = new Date();

    this.predictionService.predict(this.form).subscribe({
      next: (res) => {
        this.result = res;
        this.loading = false;
      },
      error: () => {
        this.error =
          'Não foi possível conectar à API. Verifique se o servidor está em execução em http://localhost:8000.';
        this.loading = false;
      },
    });
  }

  novaAnalise(): void {
    this.result = null;
    this.error = null;
    this.form = {
      area_pior: null!,
      textura_pior: null!,
      pontos_concavos_pior: null!,
      concavidade_pior: null!,
    };
  }

  get isMaligno(): boolean {
    return this.result?.diagnostico === 'Maligno';
  }

  get barraConfiancaWidth(): string {
    return `${this.result?.confianca ?? 0}%`;
  }

  get barraMalignoWidth(): string {
    return `${(this.result?.probabilidade_maligno ?? 0) * 100}%`;
  }

  get barraBenignoWidth(): string {
    return `${(this.result?.probabilidade_benigno ?? 0) * 100}%`;
  }
}
