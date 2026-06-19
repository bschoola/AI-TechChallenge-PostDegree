import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface PredictionInput {
  area_pior: number;
  textura_pior: number;
  pontos_concavos_pior: number;
  concavidade_pior: number;
}

export interface PredictionResponse {
  diagnostico: string;
  confianca: number;
  classe: number;
  probabilidade_maligno: number;
  probabilidade_benigno: number;
}

export interface LaudoResponse extends PredictionResponse {
  laudo: string;
}

@Injectable({ providedIn: 'root' })
export class PredictionService {
  private readonly apiUrl = 'http://localhost:8000';

  constructor(private http: HttpClient) {}

  predict(input: PredictionInput): Observable<PredictionResponse> {
    return this.http.post<PredictionResponse>(
      `${this.apiUrl}/predict/breastCancer`,
      input
    );
  }

  gerarLaudo(input: PredictionInput): Observable<LaudoResponse> {
    return this.http.post<LaudoResponse>(
      `${this.apiUrl}/predict/breastCancer/laudo`,
      input
    );
  }
}
