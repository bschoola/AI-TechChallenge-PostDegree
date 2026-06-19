import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import {
  PredictionService,
  PredictionInput,
  PredictionResponse,
  LaudoResponse,
} from './services/prediction.service';
import { MarkdownPipe } from './pipes/markdown.pipe';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, FormsModule, MarkdownPipe],
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
  laudo: string | null = null;
  loadingPredict = false;
  loadingLaudo = false;
  error: string | null = null;
  errorLaudo: string | null = null;
  dataAtual = new Date();

  constructor(private predictionService: PredictionService) {}

  submit(): void {
    this.loadingPredict = true;
    this.loadingLaudo = false;
    this.error = null;
    this.errorLaudo = null;
    this.result = null;
    this.laudo = null;
    this.dataAtual = new Date();

    // Passo 1: classificador rápido
    this.predictionService.predict(this.form).subscribe({
      next: (res) => {
        this.result = res;
        this.loadingPredict = false;

        // Passo 2: laudo via LLM (em segundo plano)
        this.loadingLaudo = true;
        this.predictionService.gerarLaudo(this.form).subscribe({
          next: (laudoRes) => {
            this.laudo = laudoRes.laudo;
            this.loadingLaudo = false;
          },
          error: () => {
            this.errorLaudo = 'Não foi possível gerar o laudo. Verifique se o serviço de IA está disponível.';
            this.loadingLaudo = false;
          },
        });
      },
      error: () => {
        this.error =
          'Não foi possível conectar à API. Verifique se o servidor está em execução em http://localhost:8000.';
        this.loadingPredict = false;
      },
    });
  }

  novaAnalise(): void {
    this.result = null;
    this.laudo = null;
    this.error = null;
    this.errorLaudo = null;
    this.loadingPredict = false;
    this.loadingLaudo = false;
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
